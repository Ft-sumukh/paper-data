import logging
import datetime
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select
from src.database.models import (
    Asset, Price, Return, BacktestRun, 
    PortfolioWeight, PortfolioReturn, Transaction, RiskMetric, Experiment
)

logger = logging.getLogger(__name__)

# Asset Operations
def get_or_create_asset(db: Session, symbol: str, name: str = None, sector: str = None, asset_class: str = "Equity") -> Asset:
    asset = db.query(Asset).filter(Asset.symbol == symbol).first()
    if not asset:
        asset = Asset(symbol=symbol, name=name or symbol, sector=sector or "Unknown", asset_class=asset_class)
        db.add(asset)
        db.commit()
        db.refresh(asset)
    return asset

def get_all_assets(db: Session) -> List[Asset]:
    return db.query(Asset).order_by(Asset.symbol).all()

# Price & Return Ingestion
def bulk_insert_prices(db: Session, clean_prices_df: pd.DataFrame, asset_metadata: List[Dict[str, str]] = None):
    """
    Inserts clean price matrices into the database, creating Asset entries if absent.
    clean_prices_df: Index = Date, Columns = Symbols
    """
    meta_dict = {a["symbol"]: a for a in (asset_metadata or [])}
    logger.info(f"Bulk inserting prices for {len(clean_prices_df.columns)} assets...")

    for symbol in clean_prices_df.columns:
        meta = meta_dict.get(symbol, {})
        asset = get_or_create_asset(
            db, 
            symbol=symbol, 
            name=meta.get("name", symbol), 
            sector=meta.get("sector", "Unknown"), 
            asset_class=meta.get("asset_class", "Equity")
        )
        
        # Check existing dates to prevent duplicate key errors
        existing_dates = set(
            d[0] for d in db.query(Price.date).filter(Price.asset_id == asset.id).all()
        )
        
        price_objects = []
        return_objects = []
        
        series = clean_prices_df[symbol].dropna()
        simple_rets = series.pct_change()
        log_rets = np.log(series / series.shift(1))
        
        for dt, price_val in series.items():
            date_val = dt.date() if isinstance(dt, pd.Timestamp) else dt
            if date_val in existing_dates:
                continue
                
            price_objects.append(Price(
                asset_id=asset.id,
                date=date_val,
                close=float(price_val),
                adj_close=float(price_val)
            ))
            
            # Simple & Log Returns
            s_ret = simple_rets.loc[dt]
            l_ret = log_rets.loc[dt]
            if not np.isnan(s_ret) and not np.isnan(l_ret):
                return_objects.append(Return(
                    asset_id=asset.id,
                    date=date_val,
                    simple_return=float(s_ret),
                    log_return=float(l_ret)
                ))
                
        if price_objects:
            db.bulk_save_objects(price_objects)
        if return_objects:
            db.bulk_save_objects(return_objects)
        db.commit()

    logger.info("Successfully populated prices and returns tables.")

def get_price_matrix(db: Session, symbols: Optional[List[str]] = None, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    """
    Returns a wide-format price DataFrame indexed by Date with assets as columns.
    """
    query = db.query(Price.date, Asset.symbol, Price.adj_close).join(Asset, Price.asset_id == Asset.id)
    if symbols:
        query = query.filter(Asset.symbol.in_(symbols))
    if start_date:
        query = query.filter(Price.date >= pd.to_datetime(start_date).date())
    if end_date:
        query = query.filter(Price.date <= pd.to_datetime(end_date).date())
        
    records = query.all()
    if not records:
        return pd.DataFrame()
        
    df = pd.DataFrame(records, columns=["date", "symbol", "adj_close"])
    pivot_df = df.pivot(index="date", columns="symbol", values="adj_close")
    pivot_df.index = pd.to_datetime(pivot_df.index)
    return pivot_df.sort_index()

# Backtest Recording
def save_backtest_run(
    db: Session,
    name: str,
    strategy: str,
    start_date: datetime.date,
    end_date: datetime.date,
    rebalance_freq: str,
    initial_capital: float,
    transaction_cost_bps: float,
    slippage_bps: float,
    weights_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    risk_metrics_dict: Dict[str, float]
) -> BacktestRun:
    """
    Persists a complete backtest run, including daily weights, returns, and summary risk metrics.
    """
    run = BacktestRun(
        name=name,
        strategy=strategy,
        start_date=start_date,
        end_date=end_date,
        rebalance_freq=rebalance_freq,
        initial_capital=initial_capital,
        transaction_cost_bps=transaction_cost_bps,
        slippage_bps=slippage_bps,
        status="COMPLETED"
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    # Map symbols to asset_ids
    assets = {a.symbol: a.id for a in db.query(Asset).all()}

    # Save weights
    weight_objs = []
    for dt, row in weights_df.iterrows():
        d_val = dt.date() if isinstance(dt, pd.Timestamp) else dt
        for sym, w in row.items():
            if sym in assets and abs(w) > 1e-6:
                weight_objs.append(PortfolioWeight(
                    backtest_run_id=run.id,
                    asset_id=assets[sym],
                    date=d_val,
                    weight=float(w)
                ))
    if weight_objs:
        db.bulk_save_objects(weight_objs)

    # Save portfolio returns
    return_objs = []
    for dt, row in returns_df.iterrows():
        d_val = dt.date() if isinstance(dt, pd.Timestamp) else dt
        return_objs.append(PortfolioReturn(
            backtest_run_id=run.id,
            date=d_val,
            gross_return=float(row.get("gross_return", 0.0)),
            net_return=float(row.get("net_return", 0.0)),
            turnover=float(row.get("turnover", 0.0)),
            transaction_cost=float(row.get("transaction_cost", 0.0)),
            portfolio_value=float(row.get("portfolio_value", initial_capital))
        ))
    if return_objs:
        db.bulk_save_objects(return_objs)

    # Save risk metrics
    metric_objs = []
    for metric_name, val in risk_metrics_dict.items():
        if isinstance(val, (int, float)) and not np.isnan(val):
            metric_objs.append(RiskMetric(
                backtest_run_id=run.id,
                metric_name=metric_name,
                metric_value=float(val),
                period_start=start_date,
                period_end=end_date
            ))
    if metric_objs:
        db.bulk_save_objects(metric_objs)

    db.commit()
    return run

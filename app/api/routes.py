import pandas as pd
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Dict, Optional, Any

from src.database.connection import get_db
from src.database.models import Asset
from src.database.crud import get_all_assets, get_price_matrix
from src.database.queries import query_asset_annual_metrics, query_strategy_comparison
from src.features.returns import calculate_simple_returns, calculate_covariance_matrix
from src.risk.metrics import calculate_all_risk_metrics
from src.optimization import get_optimizer
from src.backtesting.engine import WalkForwardBacktester
from src.regimes.detector import RollingVolatilityRegimeDetector

from app.api.schemas import (
    AssetSchema, OptimizationRequest, OptimizationResponse,
    BacktestRequest, BacktestResponse, RiskAnalysisRequest, 
    RiskAnalysisResponse, RegimeResponse
)

router = APIRouter()

@router.get("/assets", response_model=List[AssetSchema], summary="List all universe assets")
def list_assets(db: Session = Depends(get_db)):
    return get_all_assets(db)

@router.get("/prices", summary="Get historical price matrix")
def get_prices(
    symbols: Optional[str] = Query(None, description="Comma-separated ticker symbols"),
    start_date: Optional[str] = "2020-01-01",
    end_date: Optional[str] = "2024-12-31",
    db: Session = Depends(get_db)
):
    symbol_list = [s.strip() for s in symbols.split(",")] if symbols else None
    price_df = get_price_matrix(db, symbols=symbol_list, start_date=start_date, end_date=end_date)
    if price_df.empty:
        raise HTTPException(status_code=404, detail="No price data found for requested query.")
    
    return {
        "dates": [str(d.date()) for d in price_df.index],
        "symbols": list(price_df.columns),
        "data": {col: price_df[col].tolist() for col in price_df.columns}
    }

@router.get("/returns", summary="Get simple returns matrix")
def get_returns(
    symbols: Optional[str] = Query(None),
    start_date: Optional[str] = "2020-01-01",
    end_date: Optional[str] = "2024-12-31",
    db: Session = Depends(get_db)
):
    symbol_list = [s.strip() for s in symbols.split(",")] if symbols else None
    price_df = get_price_matrix(db, symbols=symbol_list, start_date=start_date, end_date=end_date)
    if price_df.empty:
        raise HTTPException(status_code=404, detail="No price data found.")
    
    rets_df = calculate_simple_returns(price_df)
    return {
        "dates": [str(d.date()) for d in rets_df.index],
        "symbols": list(rets_df.columns),
        "returns": {col: rets_df[col].tolist() for col in rets_df.columns}
    }

@router.post("/portfolio/optimize", response_model=OptimizationResponse, summary="Optimize portfolio weights")
def optimize_portfolio(req: OptimizationRequest, db: Session = Depends(get_db)):
    price_df = get_price_matrix(db, symbols=req.symbols, start_date=req.start_date, end_date=req.end_date)
    if price_df.empty or len(price_df.columns) < 2:
        raise HTTPException(status_code=400, detail="Insufficient price data for optimization.")

    rets = calculate_simple_returns(price_df)
    expected_returns = rets.mean().values * 252.0
    cov_matrix = calculate_covariance_matrix(rets, annualize=True)

    # Sector mapping from DB
    assets = db.query(Asset).filter(Asset.symbol.in_(price_df.columns)).all()
    sector_map = {a.symbol: a.sector for a in assets}

    optimizer = get_optimizer(
        strategy_name=req.strategy,
        risk_free_rate=req.risk_free_rate,
        min_weight=req.min_weight,
        max_weight=req.max_weight,
        max_sector_weight=req.max_sector_weight,
        risk_aversion=req.risk_aversion
    )

    res = optimizer.optimize(
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
        symbols=list(price_df.columns),
        sector_mapping=sector_map
    )

    risk_contrib_dict = dict(zip(price_df.columns, [float(c) for c in res.percentage_risk_contributions]))

    return OptimizationResponse(
        strategy=req.strategy,
        weights=res.symbol_weights,
        expected_return=res.expected_return,
        volatility=res.volatility,
        sharpe_ratio=res.sharpe_ratio,
        risk_contributions=risk_contrib_dict,
        converged=res.converged,
        message=res.message
    )

@router.post("/portfolio/risk", response_model=RiskAnalysisResponse, summary="Compute comprehensive risk metrics")
def calculate_risk(req: RiskAnalysisRequest):
    r_series = pd.Series(req.returns)
    b_series = pd.Series(req.benchmark_returns) if req.benchmark_returns else None
    metrics = calculate_all_risk_metrics(r_series, b_series, risk_free_rate=req.risk_free_rate)
    return RiskAnalysisResponse(metrics=metrics)

@router.post("/backtest", response_model=BacktestResponse, summary="Run walk-forward out-of-sample backtest")
def run_backtest(req: BacktestRequest, db: Session = Depends(get_db)):
    price_df = get_price_matrix(db, symbols=req.symbols)
    if price_df.empty:
        raise HTTPException(status_code=400, detail="Database contains no price data for backtest.")

    assets = db.query(Asset).all()
    sector_map = {a.symbol: a.sector for a in assets}

    backtester = WalkForwardBacktester(
        strategy_name=req.strategy,
        prices=price_df,
        symbols=req.symbols,
        benchmark_symbol=req.benchmark,
        rebalance_frequency=req.rebalance_frequency,
        estimation_window=req.estimation_window,
        initial_capital=req.initial_capital,
        transaction_cost_bps=req.transaction_cost_bps,
        slippage_bps=req.slippage_bps,
        sector_mapping=sector_map
    )
    result = backtester.run()
    perf = result["performance_df"]

    return BacktestResponse(
        summary=result["summary"],
        dates=[str(d.date()) for d in perf.index],
        portfolio_values=perf["portfolio_value"].tolist(),
        net_returns=perf["net_return"].tolist(),
        gross_returns=perf["gross_return"].tolist(),
        turnovers=perf["turnover"].tolist()
    )

@router.get("/regimes", response_model=RegimeResponse, summary="Detect market volatility regimes")
def get_regimes(benchmark: str = "SPY", window: int = 63, db: Session = Depends(get_db)):
    price_df = get_price_matrix(db, symbols=[benchmark])
    if price_df.empty:
        raise HTTPException(status_code=404, detail=f"Benchmark {benchmark} not found in database.")

    bmk_rets = calculate_simple_returns(price_df[benchmark])
    detector = RollingVolatilityRegimeDetector(window=window)
    regime_df = detector.fit_predict(bmk_rets)

    counts = regime_df["regime"].value_counts().to_dict()

    return RegimeResponse(
        dates=[str(d.date()) for d in regime_df.index],
        realized_volatility=regime_df["realized_volatility"].tolist(),
        regimes=regime_df["regime"].tolist(),
        regime_counts=counts
    )

@router.get("/metrics", summary="Get asset annual metrics table via SQL")
def get_metrics(db: Session = Depends(get_db)):
    metrics_df = query_asset_annual_metrics(db)
    return metrics_df.to_dict(orient="records")

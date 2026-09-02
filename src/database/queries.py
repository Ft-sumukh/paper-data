import pandas as pd
from sqlalchemy import text
from sqlalchemy.orm import Session
from typing import List, Dict, Any

def query_asset_annual_metrics(db: Session) -> pd.DataFrame:
    """
    Computes annualized mean return and annualized volatility per asset directly via SQL aggregations.
    """
    sql = text("""
        SELECT 
            a.symbol,
            a.sector,
            COUNT(r.simple_return) AS trading_days,
            AVG(r.simple_return) * 252.0 AS annualized_return,
            (
                SELECT SQRT(AVG(r2.simple_return * r2.simple_return) - AVG(r2.simple_return) * AVG(r2.simple_return)) * SQRT(252.0)
                FROM returns r2 WHERE r2.asset_id = a.id
            ) AS annualized_volatility
        FROM assets a
        JOIN returns r ON a.id = r.asset_id
        GROUP BY a.id, a.symbol, a.sector
        ORDER BY annualized_return DESC;
    """)
    result = db.execute(sql).fetchall()
    return pd.DataFrame(result, columns=["symbol", "sector", "trading_days", "annualized_return", "annualized_volatility"])

def query_strategy_comparison(db: Session) -> pd.DataFrame:
    """
    Aggregates overall performance and risk metrics across all backtest runs.
    """
    sql = text("""
        SELECT 
            b.id AS run_id,
            b.strategy,
            b.rebalance_freq,
            b.start_date,
            b.end_date,
            MAX(CASE WHEN m.metric_name = 'annualized_return_net' THEN m.metric_value END) AS net_annual_return,
            MAX(CASE WHEN m.metric_name = 'annualized_volatility_net' THEN m.metric_value END) AS net_annual_volatility,
            MAX(CASE WHEN m.metric_name = 'sharpe_ratio_net' THEN m.metric_value END) AS net_sharpe,
            MAX(CASE WHEN m.metric_name = 'sortino_ratio_net' THEN m.metric_value END) AS net_sortino,
            MAX(CASE WHEN m.metric_name = 'max_drawdown_net' THEN m.metric_value END) AS max_drawdown,
            MAX(CASE WHEN m.metric_name = 'var_95_historical_net' THEN m.metric_value END) AS var_95,
            MAX(CASE WHEN m.metric_name = 'cvar_95_net' THEN m.metric_value END) AS cvar_95,
            MAX(CASE WHEN m.metric_name = 'total_turnover' THEN m.metric_value END) AS total_turnover
        FROM backtest_runs b
        LEFT JOIN risk_metrics m ON b.id = m.backtest_run_id
        GROUP BY b.id, b.strategy, b.rebalance_freq, b.start_date, b.end_date
        ORDER BY net_sharpe DESC;
    """)
    result = db.execute(sql).fetchall()
    cols = [
        "run_id", "strategy", "rebalance_freq", "start_date", "end_date",
        "net_annual_return", "net_annual_volatility", "net_sharpe", "net_sortino",
        "max_drawdown", "var_95", "cvar_95", "total_turnover"
    ]
    return pd.DataFrame(result, columns=cols)

def query_sector_exposure_over_time(db: Session, backtest_run_id: int) -> pd.DataFrame:
    """
    Computes total sector exposure at each rebalance date for a backtest run.
    """
    sql = text("""
        SELECT 
            w.date,
            a.sector,
            SUM(w.weight) AS total_sector_weight
        FROM portfolio_weights w
        JOIN assets a ON w.asset_id = a.id
        WHERE w.backtest_run_id = :run_id
        GROUP BY w.date, a.sector
        ORDER BY w.date, a.sector;
    """)
    result = db.execute(sql, {"run_id": backtest_run_id}).fetchall()
    df = pd.DataFrame(result, columns=["date", "sector", "total_sector_weight"])
    if not df.empty:
        pivot_df = df.pivot(index="date", columns="sector", values="total_sector_weight").fillna(0.0)
        return pivot_df
    return df

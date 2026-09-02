import pytest
import datetime
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.database.connection import Base
from src.database.models import Asset, Price, Return, BacktestRun
from src.database.crud import (
    get_or_create_asset, get_all_assets, bulk_insert_prices, 
    get_price_matrix, save_backtest_run
)
from src.database.queries import query_asset_annual_metrics, query_strategy_comparison

@pytest.fixture
def test_db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()

def test_asset_crud(test_db):
    asset = get_or_create_asset(test_db, symbol="AAPL", name="Apple Inc.", sector="Technology")
    assert asset.id is not None
    assert asset.symbol == "AAPL"

    assets = get_all_assets(test_db)
    assert len(assets) == 1
    assert assets[0].symbol == "AAPL"

def test_bulk_insert_and_price_matrix(test_db):
    dates = pd.date_range("2023-01-02", periods=5, freq="B")
    clean_df = pd.DataFrame({
        "AAPL": [150.0, 152.0, 151.0, 153.0, 155.0],
        "MSFT": [300.0, 305.0, 302.0, 308.0, 310.0]
    }, index=dates)

    bulk_insert_prices(
        test_db, 
        clean_prices_df=clean_df, 
        asset_metadata=[
            {"symbol": "AAPL", "sector": "Technology"},
            {"symbol": "MSFT", "sector": "Technology"}
        ]
    )

    retrieved = get_price_matrix(test_db, symbols=["AAPL", "MSFT"])
    assert retrieved.shape == (5, 2)
    assert "AAPL" in retrieved.columns
    assert retrieved.loc[dates[0], "AAPL"] == 150.0

def test_save_backtest_and_queries(test_db):
    dates = pd.date_range("2023-01-02", periods=3, freq="B")
    clean_df = pd.DataFrame({"AAPL": [100, 101, 102]}, index=dates)
    bulk_insert_prices(test_db, clean_df, [{"symbol": "AAPL", "sector": "Technology"}])

    weights_df = pd.DataFrame({"AAPL": [1.0, 1.0, 1.0]}, index=dates)
    returns_df = pd.DataFrame({
        "gross_return": [0.01, 0.01, 0.01],
        "net_return": [0.009, 0.009, 0.009],
        "turnover": [0.0, 0.0, 0.0],
        "transaction_cost": [10.0, 0.0, 0.0],
        "portfolio_value": [100000.0, 100900.0, 101808.0]
    }, index=dates)

    run = save_backtest_run(
        db=test_db,
        name="Test 1/N",
        strategy="equal_weight",
        start_date=dates[0].date(),
        end_date=dates[-1].date(),
        rebalance_freq="monthly",
        initial_capital=100000.0,
        transaction_cost_bps=10.0,
        slippage_bps=5.0,
        weights_df=weights_df,
        returns_df=returns_df,
        risk_metrics_dict={"annualized_return_net": 0.12, "sharpe_ratio_net": 1.4}
    )

    assert run.id is not None
    comparison_df = query_strategy_comparison(test_db)
    assert len(comparison_df) == 1
    assert comparison_df.iloc[0]["strategy"] == "equal_weight"

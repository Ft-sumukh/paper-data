import datetime
from sqlalchemy import (
    Column, Integer, String, Float, DateTime, Date, ForeignKey, 
    Index, Text, UniqueConstraint
)
from sqlalchemy.orm import relationship
from src.database.connection import Base

def utc_now():
    return datetime.datetime.now(datetime.timezone.utc)

class Asset(Base):
    __tablename__ = "assets"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(20), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=True)
    sector = Column(String(50), index=True, nullable=True)
    asset_class = Column(String(50), default="Equity", nullable=False)
    created_at = Column(DateTime, default=utc_now)

    # Relationships
    prices = relationship("Price", back_populates="asset", cascade="all, delete-orphan")
    returns = relationship("Return", back_populates="asset", cascade="all, delete-orphan")
    weights = relationship("PortfolioWeight", back_populates="asset")
    transactions = relationship("Transaction", back_populates="asset")

    def __repr__(self):
        return f"<Asset(symbol='{self.symbol}', sector='{self.sector}')>"

class Price(Base):
    __tablename__ = "prices"

    id = Column(Integer, primary_key=True, index=True)
    asset_id = Column(Integer, ForeignKey("assets.id", ondelete="CASCADE"), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    open = Column(Float, nullable=True)
    high = Column(Float, nullable=True)
    low = Column(Float, nullable=True)
    close = Column(Float, nullable=True)
    adj_close = Column(Float, nullable=False)
    volume = Column(Float, nullable=True)

    # Relationship
    asset = relationship("Asset", back_populates="prices")

    __table_args__ = (
        UniqueConstraint("asset_id", "date", name="uix_asset_price_date"),
        Index("ix_price_asset_date", "asset_id", "date"),
    )

class Return(Base):
    __tablename__ = "returns"

    id = Column(Integer, primary_key=True, index=True)
    asset_id = Column(Integer, ForeignKey("assets.id", ondelete="CASCADE"), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    simple_return = Column(Float, nullable=False)
    log_return = Column(Float, nullable=False)

    # Relationship
    asset = relationship("Asset", back_populates="returns")

    __table_args__ = (
        UniqueConstraint("asset_id", "date", name="uix_asset_return_date"),
        Index("ix_return_asset_date", "asset_id", "date"),
    )

class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    strategy = Column(String(50), nullable=False, index=True)
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    rebalance_freq = Column(String(20), default="monthly")
    estimation_window = Column(Integer, default=252)
    initial_capital = Column(Float, default=1000000.0)
    transaction_cost_bps = Column(Float, default=10.0)
    slippage_bps = Column(Float, default=5.0)
    status = Column(String(20), default="COMPLETED")
    created_at = Column(DateTime, default=utc_now)

    # Relationships
    weights = relationship("PortfolioWeight", back_populates="backtest_run", cascade="all, delete-orphan")
    returns = relationship("PortfolioReturn", back_populates="backtest_run", cascade="all, delete-orphan")
    transactions = relationship("Transaction", back_populates="backtest_run", cascade="all, delete-orphan")
    risk_metrics = relationship("RiskMetric", back_populates="backtest_run", cascade="all, delete-orphan")

class PortfolioWeight(Base):
    __tablename__ = "portfolio_weights"

    id = Column(Integer, primary_key=True, index=True)
    backtest_run_id = Column(Integer, ForeignKey("backtest_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    asset_id = Column(Integer, ForeignKey("assets.id", ondelete="CASCADE"), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    weight = Column(Float, nullable=False)

    # Relationships
    backtest_run = relationship("BacktestRun", back_populates="weights")
    asset = relationship("Asset", back_populates="weights")

    __table_args__ = (
        Index("ix_weight_run_date_asset", "backtest_run_id", "date", "asset_id"),
    )

class PortfolioReturn(Base):
    __tablename__ = "portfolio_returns"

    id = Column(Integer, primary_key=True, index=True)
    backtest_run_id = Column(Integer, ForeignKey("backtest_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    gross_return = Column(Float, nullable=False)
    net_return = Column(Float, nullable=False)
    turnover = Column(Float, default=0.0)
    transaction_cost = Column(Float, default=0.0)
    portfolio_value = Column(Float, nullable=False)

    # Relationship
    backtest_run = relationship("BacktestRun", back_populates="returns")

    __table_args__ = (
        UniqueConstraint("backtest_run_id", "date", name="uix_run_return_date"),
        Index("ix_port_return_run_date", "backtest_run_id", "date"),
    )

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)
    backtest_run_id = Column(Integer, ForeignKey("backtest_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    asset_id = Column(Integer, ForeignKey("assets.id", ondelete="CASCADE"), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    trade_type = Column(String(10), nullable=False)
    weight_before = Column(Float, default=0.0)
    weight_after = Column(Float, default=0.0)
    trade_value = Column(Float, default=0.0)
    cost = Column(Float, default=0.0)

    # Relationships
    backtest_run = relationship("BacktestRun", back_populates="transactions")
    asset = relationship("Asset", back_populates="transactions")

class RiskMetric(Base):
    __tablename__ = "risk_metrics"

    id = Column(Integer, primary_key=True, index=True)
    backtest_run_id = Column(Integer, ForeignKey("backtest_runs.id", ondelete="CASCADE"), nullable=False, index=True)
    metric_name = Column(String(50), nullable=False, index=True)
    metric_value = Column(Float, nullable=False)
    regime = Column(String(50), default="ALL")
    period_start = Column(Date, nullable=True)
    period_end = Column(Date, nullable=True)

    # Relationship
    backtest_run = relationship("BacktestRun", back_populates="risk_metrics")

class Experiment(Base):
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, index=True)
    experiment_name = Column(String(100), unique=True, index=True, nullable=False)
    description = Column(Text, nullable=True)
    config_json = Column(Text, nullable=False)
    results_json = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utc_now)

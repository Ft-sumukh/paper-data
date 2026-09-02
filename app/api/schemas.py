from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional, Any

class AssetSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    symbol: str
    name: Optional[str] = None
    sector: Optional[str] = None
    asset_class: str = "Equity"

class PriceHistoryRequest(BaseModel):
    symbols: Optional[List[str]] = None
    start_date: Optional[str] = "2020-01-01"
    end_date: Optional[str] = "2024-12-31"

class OptimizationRequest(BaseModel):
    symbols: List[str] = Field(default=["AAPL", "MSFT", "JPM", "JNJ", "XOM", "TLT", "GLD"])
    strategy: str = Field(default="risk_parity") # equal_weight, min_variance, mean_variance, risk_parity
    risk_free_rate: float = Field(default=0.02)
    min_weight: float = Field(default=0.0)
    max_weight: float = Field(default=0.35)
    max_sector_weight: Optional[float] = Field(default=0.40)
    risk_aversion: float = Field(default=2.5)
    start_date: Optional[str] = "2022-01-01"
    end_date: Optional[str] = "2024-12-31"

class OptimizationResponse(BaseModel):
    strategy: str
    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float
    risk_contributions: Dict[str, float]
    converged: bool
    message: str

class BacktestRequest(BaseModel):
    strategy: str = Field(default="equal_weight")
    symbols: Optional[List[str]] = None
    benchmark: str = Field(default="SPY")
    rebalance_frequency: str = Field(default="monthly")
    estimation_window: int = Field(default=252)
    transaction_cost_bps: float = Field(default=10.0)
    slippage_bps: float = Field(default=5.0)
    initial_capital: float = Field(default=1000000.0)

class BacktestResponse(BaseModel):
    summary: Dict[str, Any]
    dates: List[str]
    portfolio_values: List[float]
    net_returns: List[float]
    gross_returns: List[float]
    turnovers: List[float]

class RiskAnalysisRequest(BaseModel):
    returns: List[float]
    benchmark_returns: Optional[List[float]] = None
    risk_free_rate: float = 0.02

class RiskAnalysisResponse(BaseModel):
    metrics: Dict[str, float]

class RegimeResponse(BaseModel):
    dates: List[str]
    realized_volatility: List[float]
    regimes: List[str]
    regime_counts: Dict[str, int]

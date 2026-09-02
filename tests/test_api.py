import pytest
from fastapi.testclient import TestClient
from app.api.main import app

@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"

def test_list_assets(client):
    response = client.get("/api/v1/assets")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0

def test_optimize_endpoint(client):
    payload = {
        "symbols": ["AAPL", "MSFT", "JNJ", "TLT"],
        "strategy": "risk_parity",
        "risk_free_rate": 0.02,
        "min_weight": 0.0,
        "max_weight": 0.50
    }
    response = client.post("/api/v1/portfolio/optimize", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "weights" in data
    assert "sharpe_ratio" in data
    assert "risk_contributions" in data
    assert data["converged"] is True

def test_risk_endpoint(client):
    payload = {
        "returns": [0.01, -0.005, 0.012, -0.008, 0.015, -0.002, 0.009, 0.004, -0.003, 0.008],
        "risk_free_rate": 0.02
    }
    response = client.post("/api/v1/portfolio/risk", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "cagr" in data["metrics"]
    assert "max_drawdown" in data["metrics"]

def test_regimes_endpoint(client):
    response = client.get("/api/v1/regimes?benchmark=SPY&window=63")
    assert response.status_code == 200
    data = response.json()
    assert "realized_volatility" in data
    assert "regimes" in data

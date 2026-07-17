import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

def run_backtest(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    config: Dict[str, Any]
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Executes a realistic event-driven backtest incorporating bid-ask spreads,
    trading fees, execution slippage, and confidence thresholds.
    
    predictions: array of size (N_test,) with values [0 (DOWN), 1 (STATIONARY), 2 (UP)]
                 Note: for binary targets, it may be [0 (DOWN), 1 (UP)] or [0, 1]
    probabilities: array of size (N_test, num_classes) with prediction confidences
    """
    initial_capital = config["backtest"]["initial_capital"]
    trading_fee = config["backtest"]["trading_fee"]
    slippage = config["backtest"]["slippage"]
    min_confidence = config["backtest"]["min_confidence"]
    
    # Extract bid/ask prices and convert to real units (divide by 10000 if Nasdaq integer ticks)
    # If values are already in normal float format, division won't hurt if we assume they are scaled.
    # We detect scale: if price > 1000, we divide by 10000.0, else keep as is.
    ask_prices = test_df["ask_price_1"].values
    bid_prices = test_df["bid_price_1"].values
    
    price_scale = 10000.0 if np.mean(ask_prices) > 1000 else 1.0
    ask_prices = ask_prices / price_scale
    bid_prices = bid_prices / price_scale
    mid_prices = (ask_prices + bid_prices) / 2.0
    
    n_steps = len(predictions)
    # Align prices with predictions. If predictions are shorter (due to rolling seq),
    # slice the prices from the end of the test set.
    price_offset = len(mid_prices) - n_steps
    mid_prices = mid_prices[price_offset:]
    ask_prices = ask_prices[price_offset:]
    bid_prices = bid_prices[price_offset:]
    
    # Portfolio tracking
    cash = initial_capital
    position = 0  # -1 (short), 0 (flat), 1 (long)
    portfolio_value = np.zeros(n_steps)
    transaction_costs = 0.0
    
    trades = []  # List of trades: (step, type, price, size, cost)
    gross_profits = []
    gross_losses = []
    
    last_trade_price = 0.0
    
    for t in range(n_steps):
        pred = predictions[t]
        probs = probabilities[t]
        
        # Determine prediction class confidence
        pred_confidence = probs[pred]
        
        # Decide target position
        # For binary classification (0: DOWN, 1: UP), map:
        # 0 -> DOWN (-1), 1 -> UP (1)
        # For 3-class classification (0: DOWN, 1: STATIONARY, 2: UP), map:
        # 0 -> DOWN (-1), 1 -> STATIONARY (0), 2 -> UP (1)
        if len(probs) == 2:
            # Binary class
            if pred == 1 and pred_confidence >= min_confidence:
                target_pos = 1
            elif pred == 0 and pred_confidence >= min_confidence:
                target_pos = -1
            else:
                target_pos = 0
        else:
            # 3-class
            if pred == 2 and pred_confidence >= min_confidence:
                target_pos = 1
            elif pred == 0 and pred_confidence >= min_confidence:
                target_pos = -1
            else:
                target_pos = 0
                
        # If target position changes, execute trade
        if target_pos != position:
            trade_size = target_pos - position
            
            # Determine execution price based on trade direction (incorporates spread + slippage)
            if trade_size > 0:
                # Buying: we pay the Ask price + slippage
                exec_price = ask_prices[t] * (1.0 + slippage)
            else:
                # Selling: we receive the Bid price - slippage
                exec_price = bid_prices[t] * (1.0 - slippage)
                
            trade_value = abs(trade_size) * exec_price
            fee_cost = trade_value * trading_fee
            
            # Slippage and spread cost (difference between exec price and mid price)
            spread_slippage_cost = abs(trade_size) * abs(exec_price - mid_prices[t])
            trade_cost = fee_cost + spread_slippage_cost
            transaction_costs += trade_cost
            
            # Cash flow transaction
            if trade_size > 0:
                # Buying
                cash -= (trade_size * exec_price + fee_cost)
            else:
                # Selling
                cash += (abs(trade_size) * exec_price - fee_cost)
                
            # Track trades and profits
            if position != 0:
                # Closing or reversing position: calculate realized profit/loss
                # Realized return relative to mid price
                pnl = (exec_price - last_trade_price) * position - fee_cost
                if pnl > 0:
                    gross_profits.append(pnl)
                else:
                    gross_losses.append(pnl)
                    
            trades.append({
                "step": t,
                "type": "BUY" if trade_size > 0 else "SELL",
                "price": exec_price,
                "size": abs(trade_size),
                "cost": trade_cost
            })
            
            position = target_pos
            last_trade_price = exec_price
            
        portfolio_value[t] = cash + position * mid_prices[t]

    # Calculate statistics
    total_trades = len(trades)
    gross_return = (portfolio_value[-1] + transaction_costs - initial_capital) / initial_capital
    net_return = (portfolio_value[-1] - initial_capital) / initial_capital
    
    # Returns series for Sharpe
    port_series = pd.Series(portfolio_value)
    step_returns = port_series.pct_change().dropna()
    
    # Annualized Sharpe ratio assuming 10,000 steps per day and 252 days per year
    # step Sharpe = mean / std
    if len(step_returns) > 1 and step_returns.std() > 0:
        sharpe = (step_returns.mean() / step_returns.std()) * np.sqrt(252 * 1000)
    else:
        sharpe = 0.0
        
    # Drawdowns
    cum_max = port_series.cummax()
    drawdowns = (cum_max - port_series) / cum_max
    max_drawdown = drawdowns.max()
    
    # Win rate
    total_completed_trades = len(gross_profits) + len(gross_losses)
    win_rate = len(gross_profits) / total_completed_trades if total_completed_trades > 0 else 0.0
    
    # Profit factor
    sum_gross_profits = sum(gross_profits)
    sum_gross_losses = abs(sum(gross_losses))
    profit_factor = sum_gross_profits / sum_gross_losses if sum_gross_losses > 0 else (sum_gross_profits if sum_gross_profits > 0 else 1.0)
    
    metrics = {
        "gross_return": float(gross_return),
        "net_return": float(net_return),
        "total_transaction_costs": float(transaction_costs),
        "total_trades": int(total_trades),
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(max_drawdown),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor)
    }
    
    backtest_df = pd.DataFrame({
        "mid_price": mid_prices,
        "portfolio_value": portfolio_value,
        "drawdown": drawdowns.values
    })
    
    return metrics, backtest_df

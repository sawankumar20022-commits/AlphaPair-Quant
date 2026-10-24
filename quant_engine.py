"""
AlphaPair: Statistical Arbitrage & Cointegration Engine
Mathematical implementation of Pairs Trading using:
- Engle-Granger Cointegration & Augmented Dickey-Fuller test
- Ornstein-Uhlenbeck (OU) Mean-Reversion Half-Life
- Dynamic Rolling Z-Score Normalization
- Vectorized Backtesting with Sharpe Ratio & Max Drawdown
"""

import csv
import math
import numpy as np
import matplotlib.pyplot as plt

def load_data(filepath="data/pairs_prices.csv"):
    days, asset_a, asset_b = [], [], []
    with open(filepath, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            days.append(int(row["day"]))
            asset_a.append(float(row["asset_a"]))
            asset_b.append(float(row["asset_b"]))
    return np.array(days), np.array(asset_a), np.array(asset_b)

def compute_hedge_ratio(y, x):
    """Ordinary Least Squares (OLS) to compute optimal hedge ratio beta."""
    x_mean, y_mean = np.mean(x), np.mean(y)
    beta = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    alpha = y_mean - beta * x_mean
    return alpha, beta

def test_stationarity_adf(residuals):
    """
    Simplified Augmented Dickey-Fuller test on spread residuals.
    Tests null hypothesis of a unit root (H0: non-stationary) vs stationary.
    Regression: Delta(e_t) = gamma * e_{t-1} + error
    """
    delta_res = np.diff(residuals)
    lag_res = residuals[:-1]
    
    # Regression coefficient gamma
    gamma = np.sum(lag_res * delta_res) / np.sum(lag_res ** 2)
    residuals_fit = delta_res - gamma * lag_res
    s_err = np.sqrt(np.sum(residuals_fit ** 2) / (len(delta_res) - 1))
    se_gamma = s_err / np.sqrt(np.sum(lag_res ** 2))
    
    t_stat = gamma / se_gamma
    # Critical value at 95% confidence for cointegration residual test is ~ -2.86
    is_cointegrated = t_stat < -2.86
    return t_stat, is_cointegrated, -gamma

def calculate_half_life(theta):
    """Ornstein-Uhlenbeck half-life of mean reversion: tau = ln(2) / theta."""
    if theta <= 0:
        return float('inf')
    return math.log(2) / theta

def run_pairs_strategy(asset_a, asset_b, window=20, entry_z=2.0, exit_z=0.5, stop_z=3.5):
    """Simulates statistical arbitrage strategy based on rolling Z-score."""
    alpha, beta = compute_hedge_ratio(asset_a, asset_b)
    spread = asset_a - (alpha + beta * asset_b)
    
    n = len(spread)
    z_scores = np.zeros(n)
    positions = np.zeros(n)  # +1: Long spread (Long A, Short B), -1: Short spread
    
    for i in range(window, n):
        rolling_mean = np.mean(spread[i-window:i])
        rolling_std = np.std(spread[i-window:i])
        if rolling_std > 1e-6:
            z_scores[i] = (spread[i] - rolling_mean) / rolling_std
        else:
            z_scores[i] = 0.0
            
        # Trading logic
        curr_pos = positions[i-1]
        z = z_scores[i]
        
        if curr_pos == 0:
            if z < -entry_z:
                positions[i] = 1.0   # Long spread
            elif z > entry_z:
                positions[i] = -1.0  # Short spread
        elif curr_pos == 1:
            if z >= -exit_z or z < -stop_z:
                positions[i] = 0.0   # Take profit or stop loss
            else:
                positions[i] = 1.0
        elif curr_pos == -1:
            if z <= exit_z or z > stop_z:
                positions[i] = 0.0   # Take profit or stop loss
            else:
                positions[i] = -1.0
                
    # Calculate returns
    ret_a = np.diff(asset_a, prepend=asset_a[0]) / asset_a
    ret_b = np.diff(asset_b, prepend=asset_b[0]) / asset_b
    
    # Portfolio return from spread positions (dollar-neutral)
    spread_return = positions[:-1] * (ret_a[1:] - beta * ret_b[1:])
    spread_return = np.insert(spread_return, 0, 0.0)
    
    cumulative_returns = np.cumprod(1 + spread_return) - 1.0
    
    # Performance Metrics
    daily_mean = np.mean(spread_return)
    daily_std = np.std(spread_return)
    sharpe_ratio = (daily_mean / daily_std * np.sqrt(252)) if daily_std > 0 else 0.0
    
    # Maximum Drawdown
    equity_curve = 1 + cumulative_returns
    peak = np.maximum.accumulate(equity_curve)
    drawdowns = (peak - equity_curve) / peak
    max_drawdown = np.max(drawdowns)
    
    return {
        "alpha": alpha,
        "beta": beta,
        "spread": spread,
        "z_scores": z_scores,
        "positions": positions,
        "cumulative_returns": cumulative_returns,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "final_return": cumulative_returns[-1]
    }

def plot_analytics(days, asset_a, asset_b, res, output_path="output/backtest_results.png"):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # Subplot 1: Prices & Spread
    axes[0].plot(days, asset_a, label="Asset A", color="#1f77b4", lw=1.5)
    axes[0].plot(days, asset_b, label="Asset B", color="#ff7f0e", lw=1.5)
    axes[0].set_title(f"AlphaPair Engine | Dynamic Hedge Ratio (Beta: {res['beta']:.2f})", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Price ($)")
    axes[0].legend(loc="upper left")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    
    # Subplot 2: Z-Score & Signals
    axes[1].plot(days, res["z_scores"], label="Spread Z-Score", color="#2ca02c", lw=1.2)
    axes[1].axhline(2.0, color="red", linestyle="--", alpha=0.7, label="Threshold (+/- 2.0)")
    axes[1].axhline(-2.0, color="red", linestyle="--", alpha=0.7)
    axes[1].axhline(0.0, color="gray", linestyle=":", alpha=0.7)
    axes[1].set_title("Ornstein-Uhlenbeck Mean-Reverting Spread Z-Score", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Z-Score")
    axes[1].legend(loc="upper left")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    
    # Subplot 3: Cumulative Return & Equity Curve
    axes[2].plot(days, (1 + res["cumulative_returns"]) * 100, label="Strategy Equity Curve ($100 base)", color="#9467bd", lw=1.8)
    axes[2].set_title(f"Cumulative PnL | Sharpe: {res['sharpe_ratio']:.2f} | Max DD: {res['max_drawdown']*100:.2f}%", fontsize=11, fontweight="bold")
    axes[2].set_ylabel("Portfolio Value ($)")
    axes[2].set_xlabel("Trading Days")
    axes[2].legend(loc="upper left")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    print(f"Backtest visualization saved to: {output_path}")

def main():
    print("=" * 60)
    print("  AlphaPair: Statistical Arbitrage & Cointegration Engine  ")
    print("=" * 60)
    
    days, a, b = load_data()
    alpha, beta = compute_hedge_ratio(a, b)
    spread = a - (alpha + beta * b)
    
    t_stat, is_coint, theta = test_stationarity_adf(spread)
    half_life = calculate_half_life(theta)
    
    print(f"[*] Optimal OLS Hedge Ratio (Beta) : {beta:.4f}")
    print(f"[*] Cointegration Intercept (Alpha) : {alpha:.4f}")
    print(f"[*] ADF Test t-statistic            : {t_stat:.3f} (Cointegrated: {is_coint})")
    print(f"[*] Ornstein-Uhlenbeck Half-Life    : {half_life:.1f} days")
    
    results = run_pairs_strategy(a, b)
    print("\n--- Strategy Backtest Performance ---")
    print(f"[*] Total Cumulative Return         : {results['final_return']*100:.2f}%")
    print(f"[*] Annualized Sharpe Ratio         : {results['sharpe_ratio']:.2f}")
    print(f"[*] Maximum Drawdown                : {results['max_drawdown']*100:.2f}%")
    
    plot_analytics(days, a, b, results)
    print("=" * 60)

if __name__ == "__main__":
    main()

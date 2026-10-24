import numpy as np
import csv

def generate_cointegrated_pairs(filename="data/pairs_prices.csv", n_days=252, seed=42):
    np.random.seed(seed)
    
    # Common random walk (Stochastic Trend / Brownian Motion)
    returns = np.random.normal(0.0005, 0.015, n_days)
    trend = 100 * np.exp(np.cumsum(returns))
    
    # Stationary Mean-Reverting Spread (Ornstein-Uhlenbeck process)
    theta = 0.15      # Mean reversion speed
    sigma_ou = 0.8    # Volatility of spread
    spread = np.zeros(n_days)
    for t in range(1, n_days):
        spread[t] = spread[t-1] - theta * spread[t-1] + np.random.normal(0, sigma_ou)
        
    beta = 1.45  # Cointegration Hedge Ratio
    alpha = 15.0
    
    # Asset A and Asset B
    asset_b = trend + np.random.normal(0, 1.0, n_days)
    asset_a = alpha + beta * asset_b + spread
    
    with open(filename, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["day", "asset_a", "asset_b"])
        for i in range(n_days):
            writer.writerow([i + 1, round(float(asset_a[i]), 2), round(float(asset_b[i]), 2)])
            
    print(f"Generated {n_days} days of cointegrated pair data -> {filename}")

if __name__ == "__main__":
    generate_cointegrated_pairs()

import numpy as np
import matplotlib.pyplot as plt

def simulate_trades(num_trades, win_ratio, risk_reward_ratio, initial_capital, risk_per_trade_percent):
    """
    Simulates a series of trades and calculates the cumulative return.

    Args:
        num_trades (int): The number of trades to simulate.
        win_ratio (float): The probability of winning a trade (0 to 1).
        risk_reward_ratio (float): The ratio of potential reward to risk.
        initial_capital (float): The starting capital.
        risk_per_trade_percent (float): The percentage of capital to risk on each trade.

    Returns:
        tuple: A tuple containing:
            - trade_results (list): List of individual trade outcomes (+reward or -risk).
            - cumulative_returns_percent (list): List of cumulative returns in percentage after each trade.
    """
    capital = initial_capital
    cumulative_returns = [capital]  # Start with initial capital
    trade_results = []

    for _ in range(num_trades):
        risk_amount = capital * (risk_per_trade_percent / 100.0)
        reward_amount = risk_amount * risk_reward_ratio

        if np.random.rand() < win_ratio:
            capital += reward_amount
            trade_results.append(reward_amount)
        else:
            capital -= risk_amount
            trade_results.append(-risk_amount)
        cumulative_returns.append(capital)

    cumulative_returns_percent = [(ret - initial_capital) / initial_capital * 100 for ret in cumulative_returns]
    return trade_results, cumulative_returns_percent

def plot_results(cumulative_returns_percent):
    """
    Plots the cumulative returns over trades.

    Args:
        cumulative_returns_percent (list): List of cumulative returns in percentage.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns_percent)
    plt.title('Profitability Analyzer - Cumulative Returns')
    plt.xlabel('Number of Trades')
    plt.ylabel('Cumulative Return (%)')
    plt.grid(True)
    plt.axhline(y=0, color='r', linestyle='--') # Line at 0% return
    plt.show()

if __name__ == "__main__":
    # --- User Input ---
    num_trades = 100
    win_ratio = 0.5 # 50% win rate
    risk_reward_ratio = 2.0 # 1:2 risk reward
    initial_capital = 10000
    risk_per_trade_percent = 1.0 # 1% risk per trade

    # --- Run Simulation ---
    trade_results, cumulative_returns_percent = simulate_trades(
        num_trades, win_ratio, risk_reward_ratio, initial_capital, risk_per_trade_percent
    )

    # --- Plot Results ---
    plot_results(cumulative_returns_percent)

    # --- Output Summary (Optional) ---
    final_return_percent = cumulative_returns_percent[-1]
    print(f"--- Simulation Summary ---")
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Number of Trades: {num_trades}")
    print(f"Win Ratio: {win_ratio * 100:.2f}%")
    print(f"Risk/Reward Ratio: 1:{risk_reward_ratio:.1f}")
    print(f"Risk per Trade: {risk_per_trade_percent:.2f}%")
    print(f"Final Return: {final_return_percent:.2f}%")


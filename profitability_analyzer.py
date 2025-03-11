import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import simpledialog

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

def plot_results(all_cumulative_returns_percent, average_cumulative_returns_percent, num_simulations, win_ratio, risk_reward_ratio, num_trades, risk_per_trade_percent, initial_capital):
    """
    Plots the cumulative returns over trades for multiple simulations and their average.

    Args:
        all_cumulative_returns_percent (list of lists): List of cumulative returns in percentage for each simulation.
        average_cumulative_returns_percent (list): List of average cumulative returns in percentage.
        num_simulations (int): Number of simulations run.
        win_ratio (float): Win ratio used in simulations.
        risk_reward_ratio (float): Risk/reward ratio used in simulations.
        num_trades (int): Number of trades per simulation.
        risk_per_trade_percent (float): Risk per trade percentage.
        initial_capital (float): Initial capital.
    """
    plt.figure(figsize=(12, 7))

    # Plot individual simulations in light grey
    for returns in all_cumulative_returns_percent:
        plt.plot(returns, color='lightgrey', linewidth=0.5)

    # Plot the average cumulative return in blue
    plt.plot(average_cumulative_returns_percent, color='blue', label=f'Average ({num_simulations} simulations)')

    plt.title(f'Profitability Analyzer - Cumulative Returns ({num_simulations} Simulations)')
    plt.xlabel('Number of Trades')
    plt.ylabel('Cumulative Return (%)')
    plt.grid(True)
    plt.axhline(y=0, color='r', linestyle='--') # Line at 0% return
    plt.legend()
    plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide the main tkinter window

    # --- User Input via Dialogs ---
    num_simulations = simpledialog.askinteger("Input", "Enter number of simulations:", initialvalue=5)
    num_trades = simpledialog.askinteger("Input", "Enter number of trades per simulation:", initialvalue=100)
    win_ratio_percent = simpledialog.askfloat("Input", "Enter win ratio percentage (0-100):", initialvalue=50.0)
    win_ratio = win_ratio_percent / 100.0
    risk_reward_ratio = simpledialog.askfloat("Input", "Enter risk/reward ratio (e.g., 2 for 1:2):", initialvalue=2.0)
    initial_capital = simpledialog.askfloat("Input", "Enter initial capital:", initialvalue=10000)
    risk_per_trade_percent = simpledialog.askfloat("Input", "Enter risk per trade percentage (0-100):", initialvalue=1.0)

    # --- Run Simulations ---
    all_trade_results = []
    all_cumulative_returns_percent = []

    for _ in range(num_simulations):
        trade_results, cumulative_returns_percent = simulate_trades(
            num_trades, win_ratio, risk_reward_ratio, initial_capital, risk_per_trade_percent
        )
        all_trade_results.append(trade_results)
        all_cumulative_returns_percent.append(cumulative_returns_percent)

    # --- Calculate Average Cumulative Returns ---
    max_len = max(len(returns) for returns in all_cumulative_returns_percent)
    padded_returns = [np.pad(returns, (0, max_len - len(returns)), 'constant', constant_values=np.nan) for returns in all_cumulative_returns_percent]
    average_cumulative_returns_percent = np.nanmean(padded_returns, axis=0).tolist()


    # --- Plot Results ---
    plot_results(all_cumulative_returns_percent, average_cumulative_returns_percent, num_simulations, win_ratio, risk_reward_ratio, num_trades, risk_per_trade_percent, initial_capital)

    # --- Output Summary ---
    final_returns_percent = [returns[-1] for returns in all_cumulative_returns_percent]
    average_final_return_percent = np.mean(final_returns_percent)

    print(f"--- Simulation Summary ---")
    print(f"Number of Simulations: {num_simulations}")
    print(f"Initial Capital: ${initial_capital:.2f}")
    print(f"Number of Trades per Simulation: {num_trades}")
    print(f"Win Ratio: {win_ratio * 100:.2f}%")
    print(f"Risk/Reward Ratio: 1:{risk_reward_ratio:.1f}")
    print(f"Risk per Trade: {risk_per_trade_percent:.2f}%")
    print(f"Average Final Return (over {num_simulations} simulations): {average_final_return_percent:.2f}%")

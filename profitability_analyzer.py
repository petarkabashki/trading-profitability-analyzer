import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

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

def plot_results(ax, all_cumulative_returns_percent, average_cumulative_returns_percent, num_simulations, win_ratio, risk_reward_ratio, num_trades, risk_per_trade_percent, initial_capital):
    """
    Plots the cumulative returns over trades for multiple simulations and their average.

    Args:
        ax (matplotlib.axes._axes.Axes): Matplotlib axes object to plot on.
        all_cumulative_returns_percent (list of lists): List of cumulative returns in percentage for each simulation.
        average_cumulative_returns_percent (list): List of average cumulative returns in percentage.
        num_simulations (int): Number of simulations run.
        win_ratio (float): Win ratio used in simulations.
        risk_reward_ratio (float): Risk/reward ratio used in simulations.
        num_trades (int): Number of trades per simulation.
        risk_per_trade_percent (float): Risk per trade percentage.
        initial_capital (float): Initial capital.
    """
    ax.clear() # Clear previous plot

    # Plot individual simulations in light grey
    for returns in all_cumulative_returns_percent:
        ax.plot(returns, color='lightgrey', linewidth=0.5)

    # Plot the average cumulative return in blue
    ax.plot(average_cumulative_returns_percent, color='blue', label=f'Average ({num_simulations} simulations)')

    ax.set_title(f'Profitability Analyzer - Cumulative Returns ({num_simulations} Simulations)\nWin Ratio: {win_ratio*100:.2f}%, R:R 1:{risk_reward_ratio:.1f}, Trades: {num_trades}, Risk: {risk_per_trade_percent:.2f}%')
    ax.set_xlabel('Number of Trades')
    ax.set_ylabel('Cumulative Return (%)')
    ax.grid(True)
    ax.axhline(y=0, color='r', linestyle='--') # Line at 0% return
    ax.legend()


if __name__ == "__main__":
    st.title("Interactive Profitability Analyzer")

    # Initialize session state for button click
    if 'rerun_button_clicked' not in st.session_state:
        st.session_state.rerun_button_clicked = False

    # --- Sidebar for Parameters ---
    with st.sidebar:
        st.header("Simulation Parameters")
        num_simulations = st.slider("Number of Simulations", min_value=1, max_value=500, value=100)
        num_trades = st.slider("Number of Trades per Simulation", min_value=10, max_value=2000, value=1000)
        win_ratio_percent = st.slider("Win Ratio (%)", min_value=0.0, max_value=100.0, value=40.0)
        win_ratio = win_ratio_percent / 100.0
        risk_reward_ratio = st.slider("Risk/Reward Ratio", min_value=0.1, max_value=5.0, step=0.1, value=2.0)
        initial_capital = 10000 # Fixed initial capital, could be a slider too
        risk_per_trade_percent = st.slider("Risk per Trade (%)", min_value=0.1, max_value=10.0, step=0.1, value=1.0)

        if st.button("Re-run Simulation"): # Button in sidebar to re-run with potentially changed params
            st.session_state.rerun_button_clicked = True # Set button click state

    # --- Main Panel for Plot and Summary ---
    st.header("Simulation Results")

    # --- Run Simulations --- (Run on initial load and on button click)
    all_trade_results = []
    all_cumulative_returns_percent = []

    with st.spinner('Running simulations...'):
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
    fig, ax = plt.subplots(figsize=(10, 6)) # Create plot here
    plot_results(ax, all_cumulative_returns_percent, average_cumulative_returns_percent, num_simulations, win_ratio, risk_reward_ratio, num_trades, risk_per_trade_percent, initial_capital)
    st.pyplot(fig) # Show plot in streamlit

    # --- Output Summary ---
    final_returns_percent = [returns[-1] for returns in all_cumulative_returns_percent]
    average_final_return_percent = np.mean(final_returns_percent)

    st.write("--- Simulation Summary ---")
    st.write(f"Number of Simulations: {num_simulations}")
    st.write(f"Initial Capital: ${initial_capital:.2f}")
    st.write(f"Number of Trades per Simulation: {num_trades}")
    st.write(f"Win Ratio: {win_ratio * 100:.2f}%")
    st.write(f"Risk/Reward Ratio: 1:{risk_reward_ratio:.1f}")
    st.write(f"Risk per Trade: {risk_per_trade_percent:.2f}%")
    st.write(f"Average Final Return (over {num_simulations} simulations): {average_final_return_percent:.2f}%")

    if st.session_state.rerun_button_clicked: # Check if button was clicked
        st.session_state.rerun_button_clicked = False # Reset button click state

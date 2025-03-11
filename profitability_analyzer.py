import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

import numpy as np

def simulate_trades_vectorized(num_trades, num_simulations, win_ratio, risk_reward_ratio, risk_per_trade_percent):
    """
    Simulates multiple series of trades using fully vectorized operations and computes both the cumulative returns 
    and the cumulative log returns (expressed in percentage), along with their respective averages over simulations.

    For each trade:
      - A win yields a log return of: ln(1 + (risk_reward_ratio * risk_per_trade_percent / 100))
      - A loss yields a log return of: ln(1 - (risk_per_trade_percent / 100))
    
    The cumulative log returns are computed by summing the individual trade log returns. The actual cumulative returns 
    are then derived by converting the cumulative log returns using np.expm1.
    
    The function returns:
        - all_cumulative_returns_percent (list of lists): Cumulative returns (in percent) for each simulation,
          computed as np.expm1(cumulative_log_returns) * 100.
        - average_cumulative_returns_percent (float): The average of the final cumulative returns (in percent) across simulations.
        - all_cumulative_log_returns_percent (list of lists): Cumulative log returns (in percent) for each simulation.
        - average_cumulative_log_returns_percent (float): The average of the final cumulative log returns (in percent) across simulations.
    """
    # Generate a matrix of random outcomes: rows represent simulations, columns represent trades.
    random_outcomes = np.random.rand(num_simulations, num_trades)
    
    # Determine wins (True if outcome < win_ratio) and losses.
    wins = random_outcomes < win_ratio
    
    # Calculate log returns for wins and losses.
    win_log_return = np.log(1 + risk_reward_ratio * risk_per_trade_percent / 100.0)
    loss_log_return = np.log(1 - risk_per_trade_percent / 100.0)
    
    # Compute trade log returns for each simulation.
    trade_log_returns = np.where(wins, win_log_return, loss_log_return)
    
    # Compute cumulative log returns along each simulation.
    cumulative_log_returns = np.cumsum(trade_log_returns, axis=1)
    
    # Convert cumulative log returns to percentage.
    cumulative_log_returns_percent = cumulative_log_returns * 100
    
    # Convert cumulative log returns to actual cumulative returns using np.expm1.
    cumulative_returns_percent = np.expm1(cumulative_log_returns) * 100
    
    # Compute the average of the final cumulative returns across simulations.
    average_cumulative_returns_percent = np.mean(cumulative_returns_percent[:, -1])
    # Compute the average of the final cumulative log returns (in percent) across simulations.
    average_cumulative_log_returns_percent = np.mean(cumulative_log_returns_percent[:, -1])
    
    return (cumulative_returns_percent.tolist(), 
            average_cumulative_returns_percent, 
            cumulative_log_returns_percent.tolist(), 
            average_cumulative_log_returns_percent)


def plot_results(ax, all_cumulative_returns_percent, average_cumulative_returns_percent, num_simulations, win_ratio, risk_reward_ratio, num_trades, risk_per_trade_percent):
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
    """
    ax.clear() # Clear previous plot

    # Plot individual simulations in light grey
    for returns in all_cumulative_returns_percent:
        ax.plot(returns, color='lightgrey', linewidth=0.5)

    # Plot the average cumulative return in blue
    ax.plot(average_cumulative_returns_percent, color='blue', label=f'Average ({num_simulations} simulations)')

    ax.set_title(f'Profitability Analyzer - Cumulative Returns (Log Scale) ({num_simulations} Simulations)\nWin Ratio: {win_ratio*100:.2f}%, R:R 1:{risk_reward_ratio:.1f}, Trades: {num_trades}, Risk: {risk_per_trade_percent:.2f}%')
    ax.set_xlabel('Number of Trades')
    ax.set_ylabel('Cumulative Return (%) - Log Scale') # Updated Y-axis label
    ax.grid(True)
    ax.axhline(y=0, color='r', linestyle='--') # Line at 0% return
    ax.legend()

def calculate_drawdown(cumulative_returns_percent):
    """
    Calculates the maximum drawdown from a series of cumulative returns.

    Args:
        cumulative_returns_percent (list): List of cumulative returns in percentage.

    Returns:
        float: Maximum drawdown in percentage.
    """
    peak = 0.0
    max_drawdown = 0.0
    for ret in cumulative_returns_percent:
        if ret > peak:
            peak = ret
        drawdown = (peak - ret)
        if drawdown > max_drawdown:
            max_drawdown = drawdown
    return max_drawdown


def calculate_stats(trade_results, cumulative_returns_percent, initial_capital=100): # Initial capital is fixed to 100 for % returns
    """
    Calculates performance statistics from trade results and cumulative returns.

    Args:
        trade_results (list): List of individual trade outcomes (+reward or -risk).
        cumulative_returns_percent (list): List of cumulative returns in percentage.
        initial_capital (float): Starting capital (for percentage calculations, now fixed to 100).

    Returns:
        dict: Dictionary of performance statistics.
    """
    returns = np.array(trade_results) / (initial_capital / 100.0) # Calculate returns as percentage of initial capital


    # Sharpe Ratio (assuming risk-free rate is 0)
    sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) != 0 else np.nan

    # Sortino Ratio (assuming risk-free rate is 0)
    downside_returns = returns[returns < 0]
    downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else np.nan
    sortino_ratio = np.mean(returns) / downside_deviation if downside_deviation != 0 else np.nan

    # Profit Factor
    gross_profit = np.sum(returns[returns > 0])
    gross_loss = np.abs(np.sum(returns[returns < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.nan

    # Max Drawdown
    max_drawdown_percent = calculate_drawdown(cumulative_returns_percent)


    return {
        "Final Return": f"{cumulative_returns_percent[-1]:.2f}%",
        "Sharpe Ratio": f"{sharpe_ratio:.2f}" if not np.isnan(sharpe_ratio) else "N/A",
        "Sortino Ratio": f"{sortino_ratio:.2f}" if not np.isnan(sortino_ratio) else "N/A",
        "Profit Factor": f"{profit_factor:.2f}" if not np.isnan(profit_factor) else "N/A",
        "Max Drawdown": f"{max_drawdown_percent:.2f}%"
    }


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
        risk_per_trade_percent = st.slider("Risk per Trade (%)", min_value=0.1, max_value=10.0, step=0.1, value=1.0)

        if st.button("Re-run Simulation"): # Button in sidebar to re-run with potentially changed params
            st.session_state.rerun_button_clicked = True # Set button click state

    # --- Main Panel for Plot and Summary ---
    st.header("Simulation Results")

    # --- Run Simulations --- (Run on initial load and on button click)
    all_trade_results = []
    all_cumulative_returns_percent = []
    all_max_drawdowns = [] # List to store max drawdowns for each simulation

    with st.spinner('Running simulations...'):
        all_trade_results, all_cumulative_returns_percent = simulate_trades(
            num_trades, num_simulations, win_ratio, risk_reward_ratio, risk_per_trade_percent
        )

        for returns in all_cumulative_returns_percent:
            all_max_drawdowns.append(calculate_drawdown(returns)) # Calculate and store max drawdown

    # --- Calculate Average Cumulative Returns ---
    max_len = max(len(returns) for returns in all_cumulative_returns_percent)
    padded_returns = [np.pad(returns, (0, max_len - len(returns)), 'constant', constant_values=np.nan) for returns in all_cumulative_returns_percent]
    average_cumulative_returns_percent = np.nanmean(padded_returns, axis=0).tolist()

    # --- Plot Results ---
    fig, ax = plt.subplots(figsize=(10, 6)) # Create plot here
    plot_results(ax, all_cumulative_returns_percent, average_cumulative_returns_percent, num_simulations, win_ratio, risk_reward_ratio, num_trades, risk_per_trade_percent)
    st.pyplot(fig) # Show plot in streamlit

    # --- Calculate and Display Stats Table ---
    all_stats = []
    for i in range(num_simulations):
        all_stats.append(calculate_stats(all_trade_results[i], all_cumulative_returns_percent[i]))


    average_stats = {}
    for key in all_stats[0].keys(): # Assuming all simulations have the same stats keys
        # Extract values for each stat from all simulations, convert to float, and calculate mean
        stat_values = [float(stat[key].replace('%','').replace('N/A','nan')) for stat in all_stats] # Handle % and N/A for averaging
        stat_values = [val for val in stat_values if not np.isnan(val)] # Remove NaN values for mean calculation
        if stat_values: # Check if there are valid values to average
            average_stats[key] = f"{np.mean(stat_values):.2f}{'%' if '%' in all_stats[0][key] else ''}" if '%' in all_stats[0][key] else f"{np.mean(stat_values):.2f}"
        else:
            average_stats[key] = "N/A" # If no valid values, set to N/A

    # Calculate average max drawdown
    average_stats["Average Max Drawdown"] = f"{np.mean(all_max_drawdowns):.2f}%"

    st.write("--- Average Simulation Summary ---")
    st.table(average_stats)


    if st.session_state.rerun_button_clicked: # Check if button was clicked
        st.session_state.rerun_button_clicked = False # Reset button click state

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

def simulate_trades_vectorized(num_trades, num_simulations, win_ratio, risk_reward_ratio, risk_per_trade_percent):
    """
    Simulates multiple series of trades using vectorized operations.
    Returns:
        - cumulative_returns (list of lists): Cumulative returns (simple, via np.expm1 of cumulative log returns).
        - average_cumulative_returns (np.ndarray): Average cumulative simple returns (across simulations).
        - cumulative_log_returns (list of lists): Cumulative log returns for each simulation.
        - average_cumulative_log_returns (np.ndarray): Average cumulative log returns (time series, across simulations).
        - average_cumulative_log_return (float): Average of final cumulative log returns.
        - trade_results (list of lists): Simple trade returns for each simulation.
        - trade_log_returns (list of lists): Trade log returns for each simulation.
    """
    random_outcomes = np.random.rand(num_simulations, num_trades)
    wins = random_outcomes < win_ratio

    # Compute risk and reward in simple return (decimal) terms.
    risk_amount = (risk_per_trade_percent / 100.0)
    reward_amount = risk_amount * risk_reward_ratio

    win_return = reward_amount  # e.g. 0.02 for 1% risk at 2:1 ratio
    loss_return = -risk_amount  # e.g. -0.01 for 1% risk

    trade_results_simulation = np.where(wins, win_return, loss_return)

    # Compute log returns.
    win_log_return = np.log(1 + risk_reward_ratio * risk_per_trade_percent / 100.0)
    loss_log_return = np.log(1 - risk_per_trade_percent / 100.0)
    trade_log_returns = np.where(wins, win_log_return, loss_log_return)

    # Cumulative log returns (each row = one simulation)
    cumulative_log_returns = np.cumsum(trade_log_returns, axis=1)
    # Convert to simple cumulative returns via exp transformation.
    cumulative_returns = np.expm1(cumulative_log_returns)

    average_cumulative_returns = np.nanmean(cumulative_returns, axis=0)
    average_cumulative_log_returns = np.nanmean(cumulative_log_returns, axis=0)
    average_cumulative_log_return = np.mean(cumulative_log_returns[:, -1])

    return (cumulative_returns.tolist(),
            average_cumulative_returns,
            cumulative_log_returns.tolist(),
            average_cumulative_log_returns,
            average_cumulative_log_return,
            trade_results_simulation.tolist(),
            trade_log_returns.tolist())

import numpy as np

def calculate_profit_vs_risk(num_trades, num_simulations, win_ratio, risk_reward_ratio, risk_per_trade_percents):
    """
    Calculates the profit (average cumulative log return) for different risk per trade percentages.

    Args:
        num_trades (int): Number of trades in each simulation.
        num_simulations (int): Number of simulations to run.
        win_ratio (float): Win ratio (probability of winning a trade).
        risk_reward_ratio (float): Risk-reward ratio.
        risk_per_trade_percents (list or np.ndarray): List of risk per trade percentages to test.

    Returns:
        dict: A dictionary where keys are risk_per_trade_percents and values are the corresponding average_cumulative_log_return.
    """
    profit_vs_risk = {}
    for risk_percent in risk_per_trade_percents:
        _, _, _, _, average_cumulative_log_return, _, _ = simulate_trades_vectorized(
            num_trades, num_simulations, win_ratio, risk_reward_ratio, risk_percent
        )
        profit_vs_risk[risk_percent] = average_cumulative_log_return
    return profit_vs_risk

def plot_results(ax, cumulative_returns, average_cumulative_returns, num_simulations, win_ratio, risk_reward_ratio, num_trades, risk_per_trade_percent):
    """
    Plots the cumulative returns for each simulation and the average path.
    """
    ax.clear()
    colors = plt.cm.viridis(np.linspace(0, 1, num_simulations))
    for i, returns in enumerate(cumulative_returns):
        ax.plot(np.array(returns) * 100, color=colors[i], linewidth=0.5, alpha=0.3) # Convert to percentage for plotting
    ax.plot(average_cumulative_returns * 100, color='magenta', linewidth=2.5, label=f'Average ({num_simulations} simulations)') # Convert to percentage for plotting
    ax.set_title(f'Profitability Analyzer - Cumulative Returns ({num_simulations} Simulations)\n'
                 f'Win Ratio: {win_ratio*100:.2f}%, R:R 1:{risk_reward_ratio:.1f}, Trades: {num_trades}, Risk: {risk_per_trade_percent:.2f}%')
    ax.set_xlabel('Number of Trades')
    ax.set_ylabel('Cumulative Return (%)') # Y-axis label in percentage
    ax.grid(True)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.legend()

def calculate_drawdown_log(cumulative_log_returns):
    """
    Calculates maximum drawdown from a cumulative log returns path.
    The drawdown is defined as:
        drawdown = 1 - exp(current_log - running_peak_log)
    Returns the maximum drawdown (in decimal).
    """
    L = np.array(cumulative_log_returns)
    running_max = np.maximum.accumulate(L)
    drawdowns = 1 - np.exp(L - running_max)
    return np.max(drawdowns)

def calculate_sortino_ratio_log(trade_log_returns, target=0.0):
    """
    Calculates the Sortino ratio using log returns.
    """
    returns = np.array(trade_log_returns)
    excess_returns = returns - target
    downside = excess_returns[excess_returns < 0]
    if len(downside) > 0:
        downside_deviation = np.sqrt(np.mean(downside**2))
    else:
        downside_deviation = 0.0
    if downside_deviation > 1e-9:
        return np.mean(excess_returns) / downside_deviation
    else:
        return np.nan

def calculate_stats(average_trade_results, average_trade_log_returns, average_cumulative_returns, average_cumulative_log_returns):
    """
    Computes performance statistics for the average simulation path.
    """
    simple_returns = np.array(average_trade_results)
    log_returns = np.array(average_trade_log_returns)
    sharpe_ratio = np.mean(log_returns) / np.std(log_returns) if np.std(log_returns) != 0 else np.nan
    sortino_ratio = calculate_sortino_ratio_log(log_returns)
    gross_profit = np.sum(simple_returns[simple_returns > 0])
    gross_loss = np.abs(np.sum(simple_returns[simple_returns < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.nan
    max_drawdown = calculate_drawdown_log(average_cumulative_log_returns)

    return {
        "Final Return": f"{average_cumulative_returns[-1]*100:.2f}%", # Display as percentage
        "Sharpe Ratio": f"{sharpe_ratio:.2f}" if not np.isnan(sharpe_ratio) else "N/A",
        "Sortino Ratio": f"{sortino_ratio:.2f}" if not np.isnan(sortino_ratio) else "N/A",
        "Profit Factor": f"{profit_factor:.2f}" if not np.isnan(profit_factor) else "N/A",
        "Max Drawdown": f"{max_drawdown*100:.2f}%" # Display as percentage
    }

if __name__ == "__main__":
    st.title("Interactive Profitability Analyzer")

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
        if st.button("Re-run Simulation"):
            st.session_state.rerun_button_clicked = True

    # --- Main Panel ---
    st.header("Simulation Results")

    all_trade_results = []
    all_trade_log_returns = []
    all_cumulative_returns = []
    all_cumulative_log_returns = []
    all_max_drawdowns = []

    with st.spinner('Running simulations...'):
        (all_cumulative_returns, average_cumulative_returns,
         all_cumulative_log_returns, average_cumulative_log_returns_ts,
         average_cumulative_log_return, all_trade_results, all_trade_log_returns) = simulate_trades_vectorized(
            num_trades, num_simulations, win_ratio, risk_reward_ratio, risk_per_trade_percent
        )

        # Compute max drawdown for each simulation path.
        for log_returns in all_cumulative_log_returns:
            all_max_drawdowns.append(calculate_drawdown_log(log_returns))

        # Global max drawdown: worst drawdown across all paths.
        global_max_drawdown = np.max(all_max_drawdowns)
        # Average path drawdown: max drawdown computed on the average cumulative log returns.
        average_path_drawdown = calculate_drawdown_log(average_cumulative_log_returns_ts)

    # --- Plot Results ---
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_results(ax, all_cumulative_returns, average_cumulative_returns, num_simulations, win_ratio, risk_reward_ratio, num_trades, risk_per_trade_percent)
    st.pyplot(fig)

    # --- Calculate and Display Stats Table ---
    average_trade_results = np.nanmean(np.array(all_trade_results), axis=0)
    average_trade_log_returns = np.nanmean(np.array(all_trade_log_returns), axis=0)
    average_stats = calculate_stats(average_trade_results, average_trade_log_returns, average_cumulative_returns, average_cumulative_log_returns_ts)

    # Override drawdown stats as specified:
    average_stats["Max Drawdown (Global)"] = f"{global_max_drawdown*100:.2f}%" # Display as percentage
    average_stats["Max Drawdown"] = f"{average_path_drawdown*100:.2f}%" # Display as percentage

    st.write("--- Average Simulation Summary ---")
    st.table(average_stats)

    # --- Profit vs Risk Plot ---
    risk_per_trade_percents_range = np.linspace(0.1, 10.0, num=100) # Range of risk percentages to test
    profit_vs_risk_data = calculate_profit_vs_risk(num_trades, num_simulations, win_ratio, risk_reward_ratio, risk_per_trade_percents_range)

    fig_profit_risk, ax_profit_risk = plt.subplots(figsize=(10, 6))
    ax_profit_risk.plot(profit_vs_risk_data.keys(), [val * 100 for val in profit_vs_risk_data.values()], marker='o', linestyle='-') # Convert to percentage for plotting
    ax_profit_risk.set_title('Profit vs Risk per Trade')
    ax_profit_risk.set_xlabel('Risk per Trade (%)') # X-axis label in percentage
    ax_profit_risk.set_ylabel('Average Cumulative Log Return (%)') # Y-axis label in percentage
    ax_profit_risk.grid(True)
    st.pyplot(fig_profit_risk)

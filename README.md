# Interactive Profitability Analyzer

This project is an interactive trading simulation and performance analysis tool built with Python and Streamlit. It simulates multiple trading scenarios using vectorized operations and computes key performance metrics such as cumulative returns, maximum drawdown, Sharpe ratio, Sortino ratio, and profit factor—all using log returns.

## Features

- **Vectorized Simulations:** Efficiently simulates a large number of trades and trading paths using NumPy.
- **Log Return Calculations:** Computes cumulative log returns to accurately model portfolio performance.
- **Performance Metrics:** Calculates:
  - **Cumulative Returns:** Derived from cumulative log returns.
  - **Global Maximum Drawdown:** Worst drawdown across all simulation paths.
  - **Average Maximum Drawdown:** Drawdown calculated from the average cumulative log returns path.
  - **Sharpe Ratio:** Based on log returns.
  - **Sortino Ratio:** Focused on downside risk using log returns.
  - **Profit Factor:** Ratio of gross profit to gross loss.
- **Interactive Visualization:** Uses Matplotlib to plot individual simulation paths and an average path.
- **Streamlit Integration:** Provides an intuitive interface to adjust simulation parameters and re-run simulations.

## Installation

### Requirements

- Python 3.7+
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Streamlit](https://streamlit.io/)

### Steps

1. **Clone the Repository:**

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```
Create a Virtual Environment (Optional but Recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
Install Dependencies:

```bash
pip install numpy matplotlib streamlit
```
Usage
To run the application, execute the following command in your terminal:
```bash
streamlit run your_script_name.py
```
Replace your_script_name.py with the filename of the Python script containing the project code.

## Code Overview

### Simulation

- **simulate_trades_vectorized:**  
  This function simulates multiple trading scenarios using vectorized operations. For each trade, it calculates:
  
  - **Win Log Return:**  
    \[
    l_{\text{win}} = \ln\left(1 + \frac{\text{risk\_reward\_ratio} \times \text{risk\_per\_trade\_percent}}{100}\right)
    \]
  
  - **Loss Log Return:**  
    \[
    l_{\text{loss}} = \ln\left(1 - \frac{\text{risk\_per\_trade\_percent}}{100}\right)
    \]
  
  The cumulative log return \(L_n\) for a simulation is computed as the sum of individual trade log returns:
  
  \[
  L_n = \sum_{i=1}^{n} l_i
  \]
  
  The corresponding simple cumulative return is then obtained via:
  
  \[
  R = \exp(L_n) - 1
  \]

### Visualization

- **plot_results:**  
  This function plots the cumulative returns for each simulation path along with the average cumulative return over time.

### Performance Metrics

- **calculate_drawdown_log:**  
  Computes the maximum drawdown using cumulative log returns. The drawdown at any point is defined as:
  
  \[
  \text{Drawdown} = 1 - \exp\left(L - L_{\text{peak}}\right)
  \]
  
  where:
  - \( L \) is the current cumulative log return.
  - \( L_{\text{peak}} \) is the maximum cumulative log return observed up to that point.

- **calculate_sortino_ratio_log:**  
  Computes the Sortino ratio based on log returns. First, it calculates the downside deviation by considering only returns below a target \( T \) (usually 0). For individual trade log returns \( l_i \), the downside deviation is:
  
  \[
  \sigma_{\text{down}} = \sqrt{\frac{1}{N} \sum_{l_i < T} \left(l_i - T\right)^2}
  \]
  
  The Sortino ratio is then given by:
  
  \[
  \text{Sortino Ratio} = \frac{\overline{l} - T}{\sigma_{\text{down}}}
  \]
  
  where \( \overline{l} \) is the mean of the log returns.

- **calculate_stats:**  
  This function aggregates the performance metrics for each simulation, including:
  
  - **Sharpe Ratio:**  
    \[
    \text{Sharpe Ratio} = \frac{\overline{l}}{\sigma_{l}}
    \]
    where \( \overline{l} \) is the mean log return and \( \sigma_{l} \) is its standard deviation.
  
  - **Profit Factor:**  
    The ratio of the total gains to the total losses computed from the simple returns.
  
  - **Maximum Drawdown:**  
    As computed using the `calculate_drawdown_log` function.

## Contributing
Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

## License
This project is open source and available under the MIT License.

## Acknowledgments
This tool leverages the power of vectorized computations with NumPy and the interactive capabilities of Streamlit to provide a robust analysis framework for trading simulations.
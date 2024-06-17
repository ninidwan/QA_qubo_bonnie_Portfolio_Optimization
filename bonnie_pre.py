import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from dwave.system import LeapHybridSampler

# Utility Functions
def calculate_expected_return(stock_data):
    log_returns = np.log(stock_data / stock_data.shift(1))
    return log_returns.mean() * 252

def calculate_covariance_matrix(stock_data):
    log_returns = np.log(stock_data / stock_data.shift(1))
    return log_returns.cov() * 252

def sharpe_ratio(weights, expected_return, covariance_matrix, risk_free_rate=0.0):
    portfolio_return = np.dot(weights, expected_return)
    portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))
    return (portfolio_return - risk_free_rate) / portfolio_volatility

# Load Stock Market Data
def load_stock_data(tickers, start_date, end_date):
    stock_data = yf.download(tickers, start=start_date, end=end_date)
    return stock_data['Adj Close']

# Portfolio Optimization
def portfolio_optimization(stock_data, target_return):
    expected_return = calculate_expected_return(stock_data)
    covariance_matrix = calculate_covariance_matrix(stock_data)

    num_stocks = len(stock_data.columns)
    args = (expected_return, covariance_matrix)

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
                   {'type': 'eq', 'fun': lambda weights: np.dot(weights, expected_return) - target_return})

    bounds = tuple((0, 1) for _ in range(num_stocks))
    initial_weights = num_stocks * [1. / num_stocks]

    def portfolio_volatility(weights, expected_return, covariance_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

    result = minimize(portfolio_volatility, initial_weights, args=args, method='SLSQP', bounds=bounds, constraints=constraints)

    return result.x

# QUBO Portfolio Optimization using D-Wave
def qubo_portfolio_optimization_dwave(stock_data, target_return, lambda_reg):
    expected_return = calculate_expected_return(stock_data)
    covariance_matrix = calculate_covariance_matrix(stock_data)

    num_stocks = len(stock_data.columns)
    Q = lambda_reg * covariance_matrix.values - np.outer(expected_return.values, expected_return.values)

    # Construct QUBO matrix
    qubo = {}
    for i in range(num_stocks):
        for j in range(num_stocks):
            if i == j:
                qubo[(i, j)] = Q[i, j]
            elif Q[i, j] != 0:
                qubo[(i, j)] = Q[i, j] / 2

    print("QUBO Matrix:", qubo)  # Debug: Print QUBO matrix

    # Use D-Wave's LeapHybridSampler
    sampler = LeapHybridSampler()
    response = sampler.sample_qubo(qubo)

    # Extract the solution
    sample = next(iter(response.samples()))  # Ensure we get the first sample correctly
    weights = np.array([sample.get(i, 0) for i in range(num_stocks)])  # Handle missing keys by defaulting to 0

    print("Raw Weights from D-Wave:", weights)  # Debug: Print raw weights

    # Normalize the weights, handle division by zero
    total_weight = np.sum(weights)
    if total_weight == 0:
        print("Warning: Total weight is zero. Returning equal weights.")
        weights = np.ones(num_stocks) / num_stocks
    else:
        weights = weights / total_weight

    return weights

# Main Function
if __name__ == "__main__":
    # Load stock data
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    start_date = '2020-01-01'
    end_date = '2023-01-01'
    stock_data = load_stock_data(tickers, start_date, end_date)
    stock_data.to_csv('stock_data.csv')
    print("Stock data saved to stock_data.csv")

    # Portfolio Optimization
    target_return = 0.1
    optimal_weights = portfolio_optimization(stock_data, target_return)
    print("Optimal portfolio weights:", optimal_weights)

    # QUBO Portfolio Optimization using D-Wave
    lambda_reg = 0.5
    optimal_qubo_weights = qubo_portfolio_optimization_dwave(stock_data, target_return, lambda_reg)
    print("Optimal QUBO portfolio weights using D-Wave:", optimal_qubo_weights)
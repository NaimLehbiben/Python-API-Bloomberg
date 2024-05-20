import numpy as np
import pandas as pd
import utils.config as config

class MetricsCalculator:
    def __init__(self, other_data):
        self.risk_free_rate = self._get_risk_free_rate(other_data)
        self.other_data = other_data

    def _get_risk_free_rate(self, other_data):
        if config.RISK_FREE_RATE_TICKER in other_data:
            return other_data[config.RISK_FREE_RATE_TICKER].mean() / 252  # Assuming daily rate
        else:
            raise KeyError(f"La colonne '{config.RISK_FREE_RATE_TICKER}' n'est pas présente dans other_data")

    def calculate_return(self, asset_index):
        prices = [quote.price for quote in asset_index.price_history]
        returns = pd.Series(prices).pct_change().dropna()
        if returns.empty:
            return np.nan
        trading_days_per_year = 252
        annualized_return = (1 + returns).prod()**(trading_days_per_year / len(returns)) - 1
        return annualized_return * 100  

    def calculate_volatility(self, asset_index):
        prices = [quote.price for quote in asset_index.price_history]
        returns = pd.Series(prices).pct_change().dropna()
        if returns.empty:
            return np.nan
        trading_days_per_year = 252
        annualized_volatility = returns.std() * np.sqrt(trading_days_per_year)
        return annualized_volatility * 100  

    def calculate_sharpe_ratio(self, asset_index):
        mean_return = self.calculate_return(asset_index)
        volatility = self.calculate_volatility(asset_index)
        if volatility == 0:
            return np.nan
        sharpe_ratio = (mean_return - self.risk_free_rate * 100) / volatility
        return sharpe_ratio

    def calculate_max_drawdown(self, asset_index):
        prices = [quote.price for quote in asset_index.price_history]
        cumulative_returns = (1 + pd.Series(prices).pct_change().dropna()).cumprod()
        if cumulative_returns.empty:
            return np.nan
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns / peak) - 1
        max_drawdown = drawdown.min()
        return max_drawdown * 100  
    
    def calculate_semi_variance(self, asset_index):
        prices = [quote.price for quote in asset_index.price_history]
        returns = pd.Series(prices).pct_change().dropna()
        if returns.empty:
            return np.nan
        negative_returns = returns[returns < 0]
        if negative_returns.empty:
            return np.nan
        semi_variance = np.std(negative_returns) * np.sqrt(252)
        return semi_variance * 100  

    def calculate_sortino_ratio(self, asset_index):
        mean_return = self.calculate_return(asset_index)
        semi_variance = self.calculate_semi_variance(asset_index)
        sortino_ratio = (mean_return - self.risk_free_rate * 100) / semi_variance
        return sortino_ratio

    def calculate_information_ratio(self, asset_index, benchmark_data):
        prices = [quote.price for quote in asset_index.price_history]
        returns = pd.Series(prices).pct_change().dropna()
        if returns.empty:
            return np.nan
        # Utiliser les rendements de Low Volatility comme benchmark
        benchmark_prices = [quote.price for quote in benchmark_data.price_history]
        benchmark_returns = pd.Series(benchmark_prices).pct_change().dropna()
        if benchmark_returns.empty:
            return np.nan
        annualized_benchmark_return = (1 + benchmark_returns).prod()**(252 / len(benchmark_returns)) - 1
        tracking_error = (returns - benchmark_returns.reindex(returns.index, method='ffill')).std()
        information_ratio = (self.calculate_return(asset_index) / 100 - annualized_benchmark_return) / tracking_error
        return information_ratio

    def calculate_all_metrics(self, asset_index, benchmark_data):
        return {
            'Return': self.calculate_return(asset_index),
            'Volatility': self.calculate_volatility(asset_index),
            'Sharpe Ratio': self.calculate_sharpe_ratio(asset_index),
            'Max Drawdown': self.calculate_max_drawdown(asset_index),
            'SQRT (Semi-variance)': self.calculate_semi_variance(asset_index),
            'Sortino Ratio': self.calculate_sortino_ratio(asset_index),
            'Information Ratio': self.calculate_information_ratio(asset_index, benchmark_data)
        }
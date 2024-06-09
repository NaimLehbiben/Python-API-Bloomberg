import numpy as np
import pandas as pd
from utils.utilities import Utilities

class MetricsCalculator:
    def __init__(self, other_data, risk_free_rate_ticker):
        """
        Initialize the MetricsCalculator with the given risk-free rate ticker and other data.
        """
        self.risk_free_rate_ticker = risk_free_rate_ticker
        self.risk_free_rate = self._get_risk_free_rate(other_data)
        self.other_data = other_data
        
    def _get_risk_free_rate(self, other_data):
        """
        Get the risk-free rate from the other data provided.
        """
        if self.risk_free_rate_ticker in other_data:
            return other_data[self.risk_free_rate_ticker].mean() / 252  # Assuming daily rate
        else:
            raise KeyError(f"The column '{self.risk_free_rate_ticker}' is not present in other_data")

    def calculate_return(self, asset_index):
        """
        Calculate the annualized return of the asset index.
        """
        prices = [quote.price for quote in asset_index.price_history]
        returns = pd.Series(prices).pct_change().dropna()
        if returns.empty:
            return np.nan
        trading_days_per_year = 252
        #annualized_return = (1 + returns).prod()**(trading_days_per_year / len(returns)) - 1
        annualized_return = returns.mean() * trading_days_per_year
        return annualized_return * 100  

    def calculate_volatility(self, asset_index):
        """
        Calculate the annualized volatility of the asset index.
        """
        prices = [quote.price for quote in asset_index.price_history]
        returns = pd.Series(prices).pct_change().dropna()
        if returns.empty:
            return np.nan
        trading_days_per_year = 252
        annualized_volatility = returns.std() * np.sqrt(trading_days_per_year)
        return annualized_volatility * 100  

    def calculate_sharpe_ratio(self, asset_index):
        """
        Calculate the Sharpe ratio of the asset index.
        """
        mean_return = self.calculate_return(asset_index)
        volatility = self.calculate_volatility(asset_index)
        if volatility == 0:
            return np.nan
        sharpe_ratio = (mean_return - self.risk_free_rate * 100) / volatility
        return sharpe_ratio

    def calculate_max_drawdown(self, asset_index):
        """
        Calculate the maximum drawdown of the asset index.
        """
        prices = [quote.price for quote in asset_index.price_history]
        cumulative_returns = (1 + pd.Series(prices).pct_change().dropna()).cumprod()
        if cumulative_returns.empty:
            return np.nan
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns / peak) - 1
        max_drawdown = drawdown.min()
        return max_drawdown * 100  
    
    def calculate_semi_variance(self, asset_index):
        """
        Calculate the semi-variance of the asset index.
        """
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
        """
        Calculate the Sortino ratio of the asset index.
        """
        mean_return = self.calculate_return(asset_index)
        semi_variance = self.calculate_semi_variance(asset_index)
        sortino_ratio = (mean_return - self.risk_free_rate * 100) / semi_variance
        return sortino_ratio

    def calculate_information_ratio(self, asset_index, benchmark_data):
        """
        Calculate the Information ratio of the asset index.
        """
        prices = [quote.price for quote in asset_index.price_history]
        returns = pd.Series(prices).pct_change().dropna()
        if returns.empty:
            return np.nan
        benchmark_prices = [quote.price for quote in benchmark_data.price_history]
        benchmark_returns = pd.Series(benchmark_prices).pct_change().dropna()
        if benchmark_returns.empty:
            return np.nan
        annualized_benchmark_return = (1 + benchmark_returns).prod()**(252 / len(benchmark_returns)) - 1
        tracking_error = (returns - benchmark_returns.reindex(returns.index, method='ffill')).std()
        information_ratio = (self.calculate_return(asset_index) / 100 - annualized_benchmark_return) / tracking_error
        return information_ratio

    def calculate_all_metrics(self, asset_index, benchmark_data):
        """
        Calculate all relevant metrics for the asset index.
        """
        return {
            'Return': self.calculate_return(asset_index),
            'Volatility': self.calculate_volatility(asset_index),
            'Sharpe Ratio': self.calculate_sharpe_ratio(asset_index),
            'Max Drawdown': self.calculate_max_drawdown(asset_index),
            'SQRT (Semi-variance)': self.calculate_semi_variance(asset_index),
            'Sortino Ratio': self.calculate_sortino_ratio(asset_index),
            'Information Ratio': self.calculate_information_ratio(asset_index, benchmark_data)
        }
    
    def _calc_good_bad_mkt_stats(self, asset_indices, start_date, end_date, frequency, rebalance_at, ticker):
        """
        Calculate statistics for good and bad market conditions.
        """
        index_returns = self.other_data[ticker].pct_change().dropna()
        risk_free_rate = self.other_data[self.risk_free_rate_ticker] / 100
        rebalancing_dates = Utilities.create_rebalancing_calendar(start_date, end_date, frequency, rebalance_at)
        
        deciles = ['LowVolatilityDecile', 'MidVolatilityDecile', 'HighVolatilityDecile']
        returns = {decile: asset_indices[decile].quotes_to_dataframe().pct_change().dropna() for decile in deciles}
        
        good_mkt = {decile: [] for decile in deciles}
        bad_mkt = {decile: [] for decile in deciles}
        
        for i in range(len(rebalancing_dates) - 1):
            date = rebalancing_dates[i]
            next_date = rebalancing_dates[i + 1]
            is_market_good = index_returns.loc[date:next_date].mean().iloc[0] > risk_free_rate.loc[date:next_date].mean().iloc[0]
            
            for decile in deciles:
                period_returns = returns[decile].loc[date:next_date]
                mean_return = period_returns.mean().iloc[0] * 252
                volatility = period_returns.std().iloc[0] * np.sqrt(252)
                
                if is_market_good:
                    good_mkt[decile].append((mean_return, volatility))
                else:
                    bad_mkt[decile].append((mean_return, volatility))
                    
        avg_good_mkt = self.__calculate_averages(good_mkt, deciles)
        avg_bad_mkt = self.__calculate_averages(bad_mkt, deciles)
        
        return avg_good_mkt, avg_bad_mkt
        
    def __calculate_averages(self, data, deciles):
        """
        Calculate averages for good and bad market conditions.
        """
        return {decile: (pd.Series([x[0] for x in data[decile]]).mean()* 100 if data[decile] else 0,
                         pd.Series([x[1] for x in data[decile]]).mean() * 100 if data[decile] else 0) for decile in deciles}

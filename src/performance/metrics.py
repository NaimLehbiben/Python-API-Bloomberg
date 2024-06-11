import numpy as np
import pandas as pd
from utils.utilities import Utilities
from src.strategies.strategies import LowVolatilityDecileStrategy, HighVolatilityDecileStrategy, MidVolatilityDecileStrategy
from src.strategies.estimation_and_robustness import Estimation
from src.utils.constant import SLOPE_ALPHA, REBALANCING_MOMENT

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
            return other_data[self.risk_free_rate_ticker].mean() / 252  
        else:
            raise KeyError(f"The column '{self.risk_free_rate_ticker}' is not present in other_data")

    def calculate_total_return(self, asset_index):
        """
        Calculate the total return of the asset index.
        """
        prices = [quote.price for quote in asset_index.price_history]
        total_return = (prices[-1] / prices[0] - 1) * 100
        return total_return

    def calculate_annualized_return(self, asset_index):
        """
        Calculate the annualized return of the asset index.
        """
        prices = [quote.price for quote in asset_index.price_history]
        returns = pd.Series(prices).pct_change().dropna()
        if returns.empty:
            return np.nan
        trading_days_per_year = 252
        annualized_return = returns.mean() * trading_days_per_year
        return annualized_return * 100

    def calculate_volatility(self, asset_index, period='annual'):
        """
        Calculate the volatility of the asset index.
        period: 'annual', 'monthly', 'daily'
        """
        prices = [quote.price for quote in asset_index.price_history]
        returns = pd.Series(prices).pct_change().dropna()
        if returns.empty:
            return np.nan

        if period == 'annual':
            trading_days_per_year = 252
            volatility = returns.std() * np.sqrt(trading_days_per_year)
        elif period == 'monthly':
            trading_days_per_month = 21
            volatility = returns.std() * np.sqrt(trading_days_per_month)
        elif period == 'daily':
            volatility = returns.std()
        else:
            raise ValueError("Invalid period. Choose from 'annual', 'monthly', 'daily'.")

        return volatility * 100

    def calculate_sharpe_ratio(self, asset_index):
        """
        Calculate the Sharpe ratio of the asset index.
        """
        mean_return = self.calculate_annualized_return(asset_index)
        volatility = self.calculate_volatility(asset_index, period='annual')
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
        mean_return = self.calculate_annualized_return(asset_index)
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
        information_ratio = (self.calculate_annualized_return(asset_index) / 100 - annualized_benchmark_return) / tracking_error
        return information_ratio
    
    def calculate_var(self, asset_index, confidence_level=0.95):
        """
        Calculate the historical Value at Risk (VaR) of the asset index.
        """
        prices = [quote.price for quote in asset_index.price_history]
        returns = pd.Series(prices).pct_change().dropna()
        if returns.empty:
            return np.nan
        var = np.percentile(returns, (1 - confidence_level) * 100)
        return var * 100 
    
    def calculate_all_metrics(self, asset_index, benchmark_data):
        """
        Calculate all relevant metrics for the asset index.
        """
        return {
            'Total Return': self.calculate_total_return(asset_index),
            'Annualized Return': self.calculate_annualized_return(asset_index),
            'Annualized Volatility': self.calculate_volatility(asset_index, period='annual'),
            'Monthly Volatility': self.calculate_volatility(asset_index, period='monthly'),
            'Daily Volatility': self.calculate_volatility(asset_index, period='daily'),
            'Sharpe Ratio': self.calculate_sharpe_ratio(asset_index),
            'Max Drawdown': self.calculate_max_drawdown(asset_index),
            'SQRT (Semi-variance)': self.calculate_semi_variance(asset_index),
            'Sortino Ratio': self.calculate_sortino_ratio(asset_index),
            'Information Ratio': self.calculate_information_ratio(asset_index, benchmark_data),
            'Historical VaR (95%)': self.calculate_var(asset_index, confidence_level=0.95)
        }

    def calculate_core_metrics(self, asset_index, benchmark_data):
        """
        Calculate the core metrics for the asset index.
        """
        return {
            'Total Return': self.calculate_total_return(asset_index),
            'Annualized Return': self.calculate_annualized_return(asset_index),
            'Annualized Volatility': self.calculate_volatility(asset_index, period='annual'),
            'Monthly Volatility': self.calculate_volatility(asset_index, period='monthly'),
            'Daily Volatility': self.calculate_volatility(asset_index, period='daily'),
            'Sharpe Ratio': self.calculate_sharpe_ratio(asset_index),
            'Historical VaR (95%)': self.calculate_var(asset_index, confidence_level=0.95)
        }


    def calculate_switch_performance(asset_indices, frequency):
        
        correct_switches = 0
        incorrect_switches = 0
        correct_switch_performance = 0
        incorrect_switch_performance = 0
        total_switches = 0

        if "VolatilityTiming" in asset_indices.keys():
            low_price_df = asset_indices["LowVolatilityDecile"].quotes_to_dataframe()
            high_price_df = asset_indices["HighVolatilityDecile"].quotes_to_dataframe()
            ptf_hold_dict = asset_indices["VolatilityTiming"].strategy.ptf_hold
            for date in ptf_hold_dict.keys()[1:]:
                if ptf_hold_dict[date] != "Low":
                    total_switches +=1
                    previous_date = Utilities.get_rebalancing_date(date, -1, frequency, REBALANCING_MOMENT) 
                    perf_base = (low_price_df.loc[date] / low_price_df.loc[previous_date] - 1) *100
                    perf_high = (high_price_df.loc[date] / high_price_df.loc[previous_date] - 1) *100

                    if perf_high > perf_base:
                        correct_switches +=1
                        correct_switch_performance += (perf_high - perf_base)
                    else:
                        incorrect_switches +=1
                        incorrect_switch_performance += (perf_high - perf_base)

        return {
            'Correct Switch Percentage': 100 * correct_switches / total_switches,
            'Incorrect Switch Percentage': 100 * incorrect_switches / total_switches,
            'Correct Switch Average Performance': correct_switch_performance,
            'Incorrect Switch Average Performance': incorrect_switch_performance
        }



    def _calc_good_bad_mkt_stats(self, asset_indices, start_date, end_date, frequency, rebalance_at, ticker):
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
        return {decile: (pd.Series([x[0] for x in data[decile]]).mean()* 100 if data[decile] else 0,
                         pd.Series([x[1] for x in data[decile]]).mean() * 100 if data[decile] else 0) for decile in deciles}

from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime
from src.strategies.estimation_and_robustness import Estimation
from src.utils.utilities import Utilities
import utils.constant as constant
import numpy as np

class Strategy(ABC):
    @abstractmethod
    def generate_signals(self, market_data, compositions, date, end_date, frequency, rebalance_at):
        previous_date = Utilities.get_rebalancing_date(date, sign=-1, frequency=frequency, rebalance_at=rebalance_at, step=constant.STEP_VOL)
        
        # Calculer la volatilité de chaque actif financier
        volatilities = {ticker: Utilities.calculate_past_vol(market_data[ticker], date, previous_date) for ticker in compositions}

        # Trier les actifs financiers par volatilité décroissante
        sorted_assets = sorted(volatilities.keys(), key=lambda x: volatilities[x], reverse=False)

        # Diviser les actifs financiers en déciles
        deciles = [decile.tolist() for decile in np.array_split(sorted_assets, 10)]
        if date == end_date:
            next_date = date
        else:
            next_date = Utilities.get_rebalancing_date(date, sign=1, frequency=frequency, rebalance_at=rebalance_at)
            
        return deciles, next_date, volatilities

    def _build_slope(self, market_data, low_decile, high_decile, date, frequency, rebalance_at):
        previous_date = Utilities.get_rebalancing_date(date, sign=-1, step=constant.STEP_SLOPE, frequency=frequency, rebalance_at=rebalance_at)
        low_decile_returns = Utilities.get_ptf_returns(market_data, low_decile, previous_date, date)
        high_decile_returns = Utilities.get_ptf_returns(market_data, high_decile, previous_date, date)
        return high_decile_returns - low_decile_returns

    def generate_weights(self, decile, volatilities, weights_type, market_data,date, frequency, rebalancing_at):
        if weights_type.lower() == "equally weighted":
            return {ticker: 1/len(decile) for ticker in decile}
        elif weights_type.lower() == "vol scaling":
            #sum_inv_vol = np.sum([1 / volatilities[ticker] for ticker in decile])
            #return {ticker: 1/ volatilities[ticker] / sum_inv_vol for ticker in decile}
            sum_vol = np.sum([volatilities[ticker] for ticker in decile])
            return {ticker: volatilities[ticker] / sum_vol for ticker in decile}
        elif weights_type.lower() == "max diversification" :
            return Estimation.optimize_diversification_ratio(decile, market_data, date, {ticker : 1/len(decile) for ticker in decile},
                                                             frequency,rebalancing_at)
        else:
            raise ValueError("Unknown weights type")

class LowVolatilityDecileStrategy(Strategy):
    def __init__(self, frequency, rebalance_at, weights_type):
        self.frequency = frequency
        self.rebalance_at = rebalance_at
        self.weights_type = weights_type

    def generate_signals(self, market_data: dict[str, pd.DataFrame], compositions: list[str], date: datetime, end_date) -> dict[str, float]:
        deciles, next_date, volatilities = super().generate_signals(market_data, compositions, date, end_date, self.frequency, self.rebalance_at)
        low_volatility_decile = deciles[0]
        low_volatility_decile = Utilities.check_universe(low_volatility_decile, market_data, date, next_date)
        low_volatility_decile = self.generate_weights(low_volatility_decile, volatilities, self.weights_type, market_data, date,
                                                      self.frequency, self.rebalance_at)
        return low_volatility_decile, next_date

class HighVolatilityDecileStrategy(Strategy):
    def __init__(self, frequency, rebalance_at, weights_type):
        self.frequency = frequency
        self.rebalance_at = rebalance_at
        self.weights_type = weights_type

    def generate_signals(self, market_data: dict[str, pd.DataFrame], compositions: list[str], date: datetime, end_date) -> dict[str, float]:
        deciles, next_date, volatilities = super().generate_signals(market_data, compositions, date, end_date, self.frequency, self.rebalance_at)
        high_volatility_decile = deciles[-1]
        high_volatility_decile = Utilities.check_universe(high_volatility_decile, market_data, date, next_date)
        high_volatility_decile = self.generate_weights(high_volatility_decile, volatilities, self.weights_type, market_data, date,
                                                       self.frequency, self.rebalance_at)
        return high_volatility_decile, next_date

class MidVolatilityDecileStrategy(Strategy):
    def __init__(self, frequency, rebalance_at, weights_type):
        self.frequency = frequency
        self.rebalance_at = rebalance_at
        self.weights_type = weights_type

    def generate_signals(self, market_data: dict[str, pd.DataFrame], compositions: list[str], date: datetime, end_date) -> dict[str, float]:
        deciles, next_date, volatilities = super().generate_signals(market_data, compositions, date, end_date, self.frequency, self.rebalance_at)
        mid_volatility_decile = deciles[4]
        mid_volatility_decile = Utilities.check_universe(mid_volatility_decile, market_data, date, next_date)
        mid_volatility_decile = self.generate_weights(mid_volatility_decile, volatilities, self.weights_type, market_data, date,
                                                      self.frequency, self.rebalance_at)
        return mid_volatility_decile, next_date

class VolatilityTimingStrategy(Strategy):  
    def __init__(self, start_date, frequency, rebalance_at, weights_type):
        self.ptf_hold = {Utilities.get_rebalancing_date(start_date, 1, step=0, frequency=frequency, rebalance_at=rebalance_at): 'Low'}
        self.frequency = frequency
        self.rebalance_at = rebalance_at
        self.weights_type = weights_type
    
    def generate_signals(self, market_data: dict[str, pd.DataFrame], compositions: list[str], date: datetime, end_date) -> dict[str, float]:
        low_decile, next_date = LowVolatilityDecileStrategy(self.frequency, self.rebalance_at, self.weights_type).generate_signals(market_data, compositions, date, end_date)
        high_decile, next_date = HighVolatilityDecileStrategy(self.frequency, self.rebalance_at, self.weights_type).generate_signals(market_data, compositions, date, end_date)
        slope = self._build_slope(market_data, low_decile, high_decile, date, self.frequency, self.rebalance_at)
        if date != end_date:
            if Estimation.is_slope_positive_or_negative(slope, alpha=constant.SLOPE_ALPHA, pos_or_neg='pos'):
                self.ptf_hold.update({next_date: 'High'})
            else:
                self.ptf_hold.update({next_date: 'Low'})
        if self.ptf_hold[date] == 'High':
            return high_decile, next_date
        return low_decile, next_date

class VolatilityTimingStrategy2sided(Strategy):  
    def __init__(self, start_date, frequency, rebalance_at, weights_type):
        self.ptf_hold = {Utilities.get_rebalancing_date(start_date, 1, step=0, frequency=frequency, rebalance_at=rebalance_at): 'Mid'}
        self.frequency = frequency
        self.rebalance_at = rebalance_at
        self.weights_type = weights_type
    
    def generate_signals(self, market_data: dict[str, pd.DataFrame], compositions: list[str], date: datetime, end_date) -> dict[str, float]:
        low_decile, next_date = LowVolatilityDecileStrategy(self.frequency, self.rebalance_at, self.weights_type).generate_signals(market_data, compositions, date, end_date)
        high_decile, next_date = HighVolatilityDecileStrategy(self.frequency, self.rebalance_at, self.weights_type).generate_signals(market_data, compositions, date, end_date)
        slope = self._build_slope(market_data, low_decile, high_decile, date, self.frequency, self.rebalance_at)
        if Estimation.is_slope_positive_or_negative(slope, alpha=constant.SLOPE_ALPHA, pos_or_neg='pos'):
            self.ptf_hold.update({next_date: 'High'})
        elif Estimation.is_slope_positive_or_negative(slope, alpha=constant.SLOPE_ALPHA, pos_or_neg='neg'):
            self.ptf_hold.update({next_date: 'Low'})
        else:
            self.ptf_hold.update({next_date: 'Mid'})

        if self.ptf_hold[date] == 'High':
            return high_decile, next_date
        elif self.ptf_hold[date] == 'Low':
            return low_decile, next_date    
        mid_decile, next_date = MidVolatilityDecileStrategy(self.frequency, self.rebalance_at, self.weights_type).generate_signals(market_data, compositions, date, end_date)
        return mid_decile, next_date

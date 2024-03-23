from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime

from utils.utilities import Utilities


class Strategy(ABC):

    @abstractmethod
    def generate_signals(self, data_for_signal_generation, universe_for_signal_generation, date, previous_date):
        """
        Generate trading signals based on a series of prices.

        Parameters:
        - data_for_signal_generation: An object containing the necessary data to perform the signal generation.

        Returns:
        - A dictionary with tickers as keys and signals as values.
        """
        pass


class LowVolatilityDecileStrategy(Strategy):

    def generate_signals(market_data: dict[str, pd.DataFrame], compositions : list(str), date : datetime,previous_date) -> dict[str, float]:
        """
        Generate trading signals based on the low volatility decile strategy.

        Parameters:
        - financial_assets: A list of FinancialAsset objects.

        Returns:
        - A dictionary with FinancialAsset objects as keys and weights as values.
        """
        
        # Calculer la volatilité de chaque actif financier
        volatilities = {ticker: Utilities.calculate_past_vol(market_data[ticker],date,previous_date) for ticker in compositions}
        
        # Trier les actifs financiers par volatilité croissante
        sorted_assets = sorted(volatilities.keys(), key=lambda x: volatilities[x])
        
        # Diviser les actifs financiers en déciles
        decile_size = len(sorted_assets) // 10
        deciles = [sorted_assets[i:i+decile_size] for i in range(0, len(sorted_assets), decile_size)]
        
        # Sélectionner le décile avec la volatilité la plus faible
        low_volatility_decile = deciles[0]
        
       
        return low_volatility_decile
    
    
class HighVolatilityDecileStrategy(Strategy):

    def generate_signals(market_data: dict[str, pd.DataFrame], compositions: list[str], date: datetime, previous_date) -> dict[str, float]:
        """
        Generate trading signals based on the high volatility decile strategy.

        Parameters:
        - market_data: A dictionary of DataFrame containing market data for each financial asset.
        - compositions: A list of tickers representing the financial assets to consider.
        - date: The date for which to generate the signals.

        Returns:
        - A dictionary with tickers as keys and weights as values.
        """
        # Calculer la volatilité de chaque actif financier
        volatilities = {ticker: Utilities.calculate_past_vol(market_data[ticker], date,previous_date) for ticker in compositions}

        # Trier les actifs financiers par volatilité décroissante
        sorted_assets = sorted(volatilities.keys(), key=lambda x: volatilities[x], reverse=True)

        # Diviser les actifs financiers en déciles
        decile_size = len(sorted_assets) // 10
        deciles = [sorted_assets[i:i + decile_size] for i in range(0, len(sorted_assets), decile_size)]

        # Sélectionner le décile avec la volatilité la plus élevée
        high_volatility_decile = deciles[0]

        return high_volatility_decile
    
    
class VolatilityTimingStrategy(Strategy):  
    
    def generate_signals(market_data: dict[str, pd.DataFrame], compositions: list[str], date: datetime, previous_date) -> dict[str, float]:
        
        low_decile = LowVolatilityDecileStrategy.generate_signals(market_data,compositions,date,previous_date)
        high_decile = HighVolatilityDecileStrategy.generate_signals(market_data,compositions,date,previous_date)
        
    
if __name__ == "__main__":
    
    date = datetime(2000, 2, 29)
    
    compositions = Utilities.get_data_from_pickle("composition_par_date")
    global_market_data = Utilities.get_data_from_pickle("global_market_data")
    
    
    
from datetime import datetime
import pandas as pd
from src.utils.utilities import Utilities
from src.strategies.strategies import Strategy
from src.base.quote import Quote
from src.base.position import Position
# from src.data.data_manager import DataManager
import os
from tqdm import tqdm
import utils.config as config 


class AssetIndex:
    """
    Asset index class

    Attributes:
    - name (str): The name of the index.
    - launch_date (datetime): The launch date of the index.
    - description (str): A description of the index.
    - currency (str): The currency in which the index is valued.
    - strategy (Strategy): The strategy used for rebalancing the index.
    - rebalancing_frequency (str): The frequency at which the index is rebalanced.
    - last_price (Quote): The last price quote of the index.
    - price_history (list[Quote]): A list of historical price quotes for the index.
    - current_positions (list[Position]): A list of current positions in the index.
    - historical_position (dict[datetime, list[Position]]): Historical positions of the index.
    
    """ 
    def __init__(self, launch_date : datetime, currency: str, strategy: Strategy):
        self.launch_date = launch_date
        self.currency = currency
        self.strategy = strategy
        self.last_price: Quote = None
        self.price_history: [Quote] = []
        self.current_positions: list[Position] = []
        self.historical_position: dict[datetime, list[Position]] = {}
        
        
    def rebalance_portfolio(self, date: datetime, end_date, global_market_data: dict[str, pd.DataFrame] = None, 
                            universe : dict[str, list[str]] = None):
        """
        Rebalance the portfolio based on the given date and universe of assets.
    
        Parameters:
        - date (datetime): The date for rebalancing the portfolio.
        - end_date (datetime): The end date for rebalancing.
        - global_market_data (Dict[str, pd.DataFrame], optional): Dictionary containing market data for each ticker.
        - universe (Dict[str, List[str]], optional): Dictionary containing the universe of assets.
    
        Returns:
        None
        """
        
        universe, next_date = self.strategy.generate_signals(global_market_data,universe, date, end_date)
        
        
        self.update_historical_prices(universe,global_market_data,date, next_date)

        
        self.last_price = self.price_history[-1]

        self.update_current_and_historical_positions(date, universe)
  
    
    def update_price_history_from_list(self, new_quotes: list[Quote]):
        """
        Update the price history from a list of new quotes.

        Parameters:
        - new_quotes (list[Quote]): The list of new quotes to be added to the price history.

        """
        existing_dates = [quote.date for quote in self.price_history]

        for new_quote in new_quotes:
            if new_quote.date not in existing_dates:
                self.price_history.append(new_quote)
                existing_dates.append(new_quote.date)

        self.price_history = sorted(self.price_history, key=lambda x: x.date)   
    
    
    def quotes_to_dataframe(self) -> pd.DataFrame:
        """
        Convert the price history to a pandas DataFrame.
    
        Returns:
        pd.DataFrame: DataFrame containing the price history with the Date as index and 'Price' as a column.
        """
        if not self.price_history:
            return None
    
        dates = [quote.date for quote in self.price_history]
        prices = [quote.price for quote in self.price_history]
    
        df = pd.DataFrame({'Price': prices }, index=dates)
        df.index.name = 'Date'
        
        return df
    
    def get_quote_history_from_df(self, data: pd.DataFrame)-> list[Quote]:
        """
        Convert historical price data from a DataFrame to a list of Quote objects.

        Parameters:
        - data (pd.DataFrame): The DataFrame containing historical price data.

        Returns:
        list[Quote]: List of Quote objects representing historical prices.

        """
        dates = data.index.tolist()
        prices = data.iloc[:,0].tolist()
        return [Quote(date,price) for date,price in zip(dates,prices)]
    
    def get_last_track(self) -> float:
        """
        Gets the last track value of the index.

        Returns:
        float: The last track value.
        """
        if not self.price_history:
            return 100
        else:
            self.price_history = sorted(self.price_history, key=lambda x: x.date)
            return self.price_history[-1].price
    

    
    def update_historical_prices(self, new_weights, market_data, date, next_date):
        """
        Update the historical prices based on the new weights and market data.
    
        Parameters:
        - new_weights (dict): Dictionary containing new weights for each asset.
        - market_data (dict): Dictionary containing market data for each ticker.
        - date (datetime): The start date for updating historical prices.
        - next_date (datetime): The end date for updating historical prices.
    
        Returns:
        None
        """
        price_history_df = pd.DataFrame()
        return_history_df = pd.DataFrame()
        last_track = self.get_last_track()
        
        weighted_returns = {}
        for ticker in new_weights.keys():
            
            date_index = market_data[ticker].index.get_loc(date)-1
            next_date_index = market_data[ticker].index.get_loc(next_date)
            if date==next_date:
                next_date_index+=1
            asset_price_history = market_data[ticker].iloc[date_index:next_date_index]
            weighted_returns[ticker]= asset_price_history.iloc[:,0].pct_change().dropna() * new_weights[ticker]
      
        
        price_history_df = pd.DataFrame(weighted_returns)
        return_history_df["Rend"] = price_history_df.iloc[:, 1:].sum(axis=1)
        
        if not self.price_history:
           return_history_df.iloc[0,0]=0 
        price_history_df = pd.DataFrame({"Price": (1 + return_history_df['Rend']).cumprod() * last_track})
        
        
        # updating the historical track record
        new_quotes = self.get_quote_history_from_df(price_history_df)  
        self.update_price_history_from_list(new_quotes)

    
    def update_current_and_historical_positions(self,date:datetime, weights : dict[str, float]):  
        """
        Update current and historical positions based on the given date and weights.
    
        Parameters:
        - date (datetime): The date for updating positions.
        - weights (Dict[str, float]): Dictionary containing weights for each asset.
    
        Returns:
        None
        """       
        positions = [Position(ticker, weights[ticker]) for ticker in weights.keys()]    
        self.current_positions = positions
       
        if date not in self.historical_position:
            self.historical_position[date] = positions

        
    def get_port_file(self, ptf_name):

        data = []
        
        for date, positions in self.historical_position.items():
            for position in positions:
                data.append({
                    "Portfolio Name": ptf_name,
                    "Ticker": position.ticker,
                    "Weight": position.weight,
                    "Date": date
                })
        
        pd.DataFrame(data).to_csv( os.path.dirname(__file__).replace("src\\backtester", "data") 
                                  + "\\port.csv", sep=';', index=False, decimal=',')



class BackTesting:
      """
      Class for backtesting a asset index.
      """
      @staticmethod
      def start(params : dict):
          """
          Starts the backtesting process based on the provided parameters.

          Parameters:
          - params (dict): A dictionary containing the parameters for backtesting.
        
          Returns:
          AssetIndex: An instance of the AssetIndex class representing the backtested asset index or strategy.
          """
          
          start_date = params.get("start_date", None)
          end_date = params.get("end_date", None)
          currency = params.get("currency", None)
          ticker = params.get("ticker", None)
          use_pickle_universe = params.get("use_pickle_universe", None)
          strategy = params.get("strategy", None)
          
          rebalancing_calendar = Utilities.create_rebalancing_calendar(start_date, end_date)
          
          if not use_pickle_universe:  
              print("blapi not available on this pc")
              # compositions, global_market_data = DataManager.fetch_backtest_data(start_date, end_date, ticker, currency)                                                     
          else:
              compositions = Utilities.get_data_from_pickle("composition_par_date")
              global_market_data = Utilities.get_data_from_pickle("global_market_data")

          tracker = AssetIndex(rebalancing_calendar[0], currency, strategy)    
         
            
          end_date = rebalancing_calendar[-1]
          # Utiliser tqdm pour ajouter une barre de progression
          for date in rebalancing_calendar:
            print(f"Processing date: {date}")
            tracker.rebalance_portfolio(date, end_date, global_market_data, compositions[date])
              
          return tracker
          


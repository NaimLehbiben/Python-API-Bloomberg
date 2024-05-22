from pandas_market_calendars import get_calendar
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import pandas_market_calendars as mcal
import utils.config as config 

class Utilities:
    """
    Utility class
    """

    @staticmethod
    def create_rebalancing_calendar(start_date: datetime, end_date: datetime, frequency: str = config.REBALANCING_FREQUENCY, rebalance_at: str = config.REBALANCING_MOMENT):
        """
        Create a rebalancing calendar with business days between the start and end date, based on specified frequency and rebalancing point.

        Parameters:
        - start_date (datetime): The start date of the calendar.
        - end_date (datetime): The end date of the calendar.
        - frequency (str): The frequency of rebalancing ('monthly', 'quarterly', 'semiannually', 'annually').
        - rebalance_at (str): When to rebalance ('start' or 'end').

        Returns:
        list[datetime]: A list of rebalancing dates from start_date to end_date.
        Raises:
        ValueError: If start_date is after end_date, frequency is invalid, or rebalance_at is invalid.
        """
        if start_date > end_date:
            raise ValueError("start_date must be before end_date.")
        if frequency not in ['monthly', 'quarterly', 'semiannually', 'annually']:
            raise ValueError("Invalid frequency. Choose from 'monthly', 'quarterly', 'semiannually', 'annually'.")
        if rebalance_at not in ['start', 'end']:
            raise ValueError("Invalid rebalance_at. Choose from 'start' or 'end'.")

        nyse = mcal.get_calendar('NYSE')
        valid_dates = nyse.valid_days(start_date=start_date, end_date=end_date)
        rebalnce_dates = []
        

        if frequency == 'monthly':
            for i, date in enumerate(valid_dates[:-1]):
                if rebalance_at == 'end' and valid_dates[i+1].month != date.month:
                    rebalnce_dates.append(date.to_pydatetime().date())
                elif rebalance_at == 'start' and valid_dates[i].month != valid_dates[i-1].month:
                    rebalnce_dates.append(date.to_pydatetime().date())
        elif frequency == 'quarterly':
            for i, date in enumerate(valid_dates[:-1]):
                if rebalance_at == 'end' and (valid_dates[i+1].month - 1) // 3 != (date.month - 1) // 3:
                    rebalnce_dates.append(date.to_pydatetime().date())
                elif rebalance_at == 'start' and (valid_dates[i].month - 1) // 3 != (valid_dates[i-1].month - 1) // 3:
                    rebalnce_dates.append(date.to_pydatetime().date())
        elif frequency == 'semiannually':
            for i, date in enumerate(valid_dates[:-1]):
                if rebalance_at == 'end' and (valid_dates[i+1].month - 1) // 6 != (date.month - 1) // 6:
                    rebalnce_dates.append(date.to_pydatetime().date())
                elif rebalance_at == 'start' and (valid_dates[i].month - 1) // 6 != (valid_dates[i-1].month - 1) // 6:
                    rebalnce_dates.append(date.to_pydatetime().date())
        elif frequency == 'annually':
            for i, date in enumerate(valid_dates[:-1]):
                if rebalance_at == 'end' and valid_dates[i+1].year != date.year:
                    rebalnce_dates.append(date.to_pydatetime().date())
                elif rebalance_at == 'start' and valid_dates[i].year != valid_dates[i-1].year:
                    rebalnce_dates.append(date.to_pydatetime().date())

        if valid_dates.empty:
            rebalnce_dates.append(valid_dates[-1].date())
        
        return rebalnce_dates

    @staticmethod
    def get_rebalancing_date(date, sign, rebalance_at = config.REBALANCING_MOMENT, step = None):
        """
        Get the rebalancing date after a given step, adjusted for business days.

        Parameters:
        - date (datetime): The initial date.
        - step (int): The number of months to step forward.
        - rebalance_at (str): When to rebalance ('start' or 'end').

        Returns:
        datetime: The rebalancing date adjusted to the nearest business day.
        """
        if step is None :
            steps_dict = {'monthly' : 1, 'quarterly' : 3, 'semiannually':6, 'annually' : 12}
            step = steps_dict[config.REBALANCING_FREQUENCY]
            
        if rebalance_at == 'end':
            rebalancing_date = date + pd.DateOffset(months=step*sign)
            rebalancing_date = rebalancing_date + pd.offsets.MonthEnd(0)
        else:
            rebalancing_date = date + pd.DateOffset(months=step*sign)
            rebalancing_date = rebalancing_date - pd.offsets.MonthBegin(0) + pd.DateOffset(days=1)

        rebalancing_date = rebalancing_date.date()
        nyse = mcal.get_calendar('NYSE')
        valid_days_index = nyse.valid_days(start_date=(rebalancing_date - pd.Timedelta(days=3)), end_date=rebalancing_date)
        valid_days_list = [date.to_pydatetime().date() for date in valid_days_index]

        if not valid_days_list:
            return rebalancing_date

        while rebalancing_date not in valid_days_list:
            rebalancing_date -= pd.Timedelta(days=1)

        return rebalancing_date

    @staticmethod
    def check_universe(universe, market_data, date, next_date):
        universe = [ticker for ticker in universe if Utilities.check_data_between_dates(market_data[ticker], date, next_date)]
        if not universe:
            raise Exception("Investment universe is empty!")
        return universe

    @staticmethod
    def check_data_between_dates(df, start_date, end_date):
        if start_date not in df.index or end_date not in df.index:
            return False
        if start_date == end_date:
            return True
        data_subset = df.loc[(df.index > start_date) & (df.index < end_date)]
        if not data_subset.empty:
            return True
        return False

    @staticmethod
    def calculate_past_vol(price_history: pd.DataFrame, date: datetime, previous_date) -> float:
        price_history = price_history.loc[previous_date:date]
        returns = price_history.iloc[:, 0].pct_change().dropna()
        volatility = np.std(returns)
        return volatility

    @staticmethod
    def get_ptf_returns(data, tickers, start_date, end_date):
        selected_data = pd.concat([data[ticker].loc[start_date:end_date] for ticker in tickers.keys() if ticker in data], axis=1)
        returns = selected_data.pct_change(fill_method=None)
        weighted_returns = returns.apply(lambda col: col * tickers.get(col.name, 0.0))
        mean_weighted_returns_by_date = weighted_returns.dropna().sum(axis=1)
        return mean_weighted_returns_by_date.sort_index()

    @staticmethod
    def get_data_from_pickle(file_name: str, folder_subpath :str = None):
        if folder_subpath is None:
            file_path = os.path.join(os.path.dirname(__file__).replace("src\\utils", "data"), file_name + ".pkl")
        else:
            file_path = os.path.join(os.path.dirname(__file__).replace("src\\utils", "data\\"), folder_subpath + "\\",file_name + ".pkl")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data

    @staticmethod
    def save_data_to_pickle(data, file_name, folder_subpath :str = None):
        if folder_subpath is None:
            file_path = os.path.join(os.path.dirname(__file__).replace("src\\utils", "data"), file_name + ".pkl")
        else:
            file_path = os.path.join(os.path.dirname(__file__).replace("src\\utils", "data\\"), folder_subpath + "\\",file_name + ".pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load_asset_indices(names_list: list[str], folder_subpath :str):
        asset_indices = {}
        for name in names_list:
            asset_indices.update({name: Utilities.get_data_from_pickle(name, folder_subpath)})
        return asset_indices

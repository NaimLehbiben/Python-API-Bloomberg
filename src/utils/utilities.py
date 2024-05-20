from pandas_market_calendars import get_calendar
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import pandas_market_calendars as mcal

class Utilities:
    """
    Utility class
    """

    @staticmethod
    def create_rebalancing_calendar(start_date: datetime, end_date: datetime):
        """
        Create a rebalancing calendar with business days between the start and end date.

        Parameters:
        - start_date (datetime): The start date of the calendar.
        - end_date (datetime): The end date of the calendar.

        Returns:
        list[datetime]: A list of rebalancing dates from start_date to end_date
        Raises:
        ValueError: If start_date is after end_date.
        """
        nyse = mcal.get_calendar('NYSE')
        valid_dates = nyse.valid_days(start_date=start_date, end_date=end_date)
        rebalnce_dates = [date.to_pydatetime().date() for i, date in enumerate(valid_dates[:-1]) if valid_dates[i+1].month != date.month]
        rebalnce_dates.append(valid_dates[-1].date())
        return rebalnce_dates

    @staticmethod
    def get_rebalancing_date(date, step):
        rebalancing_date = date + pd.DateOffset(months=step)
        rebalancing_date = rebalancing_date + pd.offsets.MonthEnd(0)
        rebalancing_date = rebalancing_date.date()
        nyse = get_calendar('XNYS')
        valid_days_index = nyse.valid_days(start_date=(rebalancing_date-pd.Timedelta(days=3)), end_date=rebalancing_date)
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
    def get_data_from_pickle(file_name: str):
        file_path = os.path.join(os.path.dirname(__file__).replace("src\\utils", "data"), file_name + ".pkl")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data

    @staticmethod
    def save_data_to_pickle(data, file_name):
        file_path = os.path.join(os.path.dirname(__file__).replace("src\\utils", "data"), file_name + ".pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load_trackers(names_list: list[str]):
        trackers = {}
        for name in names_list:
            trackers.update({name: Utilities.get_data_from_pickle(name)})
        return trackers

    @staticmethod
    def load_data(file_name):
        file_path = os.path.join(os.path.dirname(__file__).replace("src\\utils", "data"), file_name + ".pkl")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data

from pandas_market_calendars import get_calendar
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import pandas_market_calendars as mcal
import utils.constant as constant

class Utilities:
    """
    Classe utilitaire
    """

    @staticmethod
    def create_rebalancing_calendar(start_date: datetime, end_date: datetime, frequency: str, rebalance_at: str):
        """
        Crée un calendrier de rebalancement basé sur les business dates du NYSE (New York Stock Exchange) 
        entre start_date et end_date.

        Paramètres:
        - start_date (datetime): La date de début.
        - end_date (datetime): La date de fin.
        - frequency (str): La fréquence du rééquilibrage. Doit être une des valeurs suivantes : 'monthly', 'quarterly', 'semiannually', 'annually'.
        - rebalance_at (str): Indique si le rééquilibrage doit avoir lieu au début ou à la fin de la période. Doit être 'start' ou 'end'.

        Exceptions:
        - ValueError: Si start_date est après end_date.
        - ValueError: Si frequency n'est pas parmi les valeurs acceptées.
        - ValueError: Si rebalance_at n'est pas parmi les valeurs acceptées.

        Retourne:
        - list: Une liste de dates de rebalancement.
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
                if rebalance_at == 'end' and valid_dates[i + 1].month != date.month:
                    rebalnce_dates.append(date.to_pydatetime().date())
                elif rebalance_at == 'start' and valid_dates[i].month != valid_dates[i - 1].month:
                    rebalnce_dates.append(date.to_pydatetime().date())
        elif frequency == 'quarterly':
            for i, date in enumerate(valid_dates[:-1]):
                if rebalance_at == 'end' and (valid_dates[i + 1].month - 1) // 3 != (date.month - 1) // 3:
                    rebalnce_dates.append(date.to_pydatetime().date())
                elif rebalance_at == 'start' and (valid_dates[i].month - 1) // 3 != (valid_dates[i - 1].month - 1) // 3:
                    rebalnce_dates.append(date.to_pydatetime().date())
        elif frequency == 'semiannually':
            for i, date in enumerate(valid_dates[:-1]):
                if rebalance_at == 'end' and (valid_dates[i + 1].month - 1) // 6 != (date.month - 1) // 6:
                    rebalnce_dates.append(date.to_pydatetime().date())
                elif rebalance_at == 'start' and (valid_dates[i].month - 1) // 6 != (valid_dates[i - 1].month - 1) // 6:
                    rebalnce_dates.append(date.to_pydatetime().date())
        elif frequency == 'annually':
            for i, date in enumerate(valid_dates[:-1]):
                if rebalance_at == 'end' and valid_dates[i + 1].year != date.year:
                    rebalnce_dates.append(date.to_pydatetime().date())
                elif rebalance_at == 'start' and valid_dates[i].year != valid_dates[i - 1].year:
                    rebalnce_dates.append(date.to_pydatetime().date())

        if valid_dates.empty:
            rebalnce_dates.append(valid_dates[-1].date())

        return rebalnce_dates

    @staticmethod
    def get_rebalancing_date(date, sign, frequency, rebalance_at, step=None):
        """
        Renoie la date de rebalancement à partir d'une date donnée.

        Paramètres:
        - date (datetime): La date de référence pour le calcul.
        - sign (int): Indicateur de direction (1 pour avancer dans le temps, -1 pour reculer).
        - frequency (str): La fréquence du rééquilibrage (par exemple, 'monthly', 'quarterly', 'semiannually', 'annually').
        - rebalance_at (str): Indique si le rééquilibrage doit avoir lieu au début ou à la fin de la période.
        - step (int, optionnel): Nombre de périodes à avancer/reculer. Si None, utilise les valeurs par défaut pour chaque fréquence.

        Retourne:
        - datetime.date: La date de rebalancement valide.
        """
        if step is None:
            steps_dict = {'monthly': 1, 'quarterly': 3, 'semiannually': 6, 'annually': 12}
            step = steps_dict[frequency]

        if rebalance_at == 'end':
            rebalancing_date = date + pd.DateOffset(months=step * sign)
            rebalancing_date = rebalancing_date + pd.offsets.MonthEnd(0)
        else:
            rebalancing_date = date + pd.DateOffset(months=step * sign)
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
        """
        Vérifie que tous les tickers de l'univers ont des données disponibles entre deux dates.

        Paramètres:
        - universe (list): Liste des tickers à vérifier.
        - market_data (dict): Dictionnaire contenant les données de marché.
        - date (datetime): Date de début de la vérification.
        - next_date (datetime): Date de fin de la vérification.

        Retourne:
        - list: Liste des tickers avec des données valides entre les deux dates.

        Exceptions:
        - Exception: Si l'univers d'investissement est vide après vérification.
        """
        universe = [ticker for ticker in universe if Utilities.check_data_between_dates(market_data[ticker], date, next_date)]
        if not universe:
            raise Exception("Investment universe is empty!")
        return universe

    @staticmethod
    def check_data_between_dates(df, start_date, end_date):
        """
        Vérifie si des données existent entre deux dates dans un DataFrame.

        Paramètres:
        - df (pd.DataFrame): DataFrame contenant les données de marché.
        - start_date (datetime): Date de début.
        - end_date (datetime): Date de fin.

        Retourne:
        - bool: True si des données existent entre les deux dates, sinon False.
        """
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
        """
        Calcule la volatilité historique entre deux dates.

        Paramètres:
        - price_history (pd.DataFrame): Historique des prix.
        - date (datetime): Date de fin pour le calcul.
        - previous_date (datetime): Date de début pour le calcul.

        Retourne:
        - float: La volatilité calculée.
        """
        price_history = price_history.loc[previous_date:date]
        returns = price_history.iloc[:, 0].pct_change().dropna()
        volatility = np.std(returns)
        return volatility

    @staticmethod
    def get_ptf_returns(data, tickers, start_date, end_date):
        """
        Calcule les rendements pondérés d'un portefeuille entre deux dates.

        Paramètres:
        - data (dict): Dictionnaire contenant les données de marché pour chaque ticker.
        - tickers (dict): Dictionnaire des tickers et de leurs pondérations dans le portefeuille.
        - start_date (datetime): Date de début pour le calcul des rendements.
        - end_date (datetime): Date de fin pour le calcul des rendements.

        Retourne:
        - pd.Series: Série des rendements pondérés par date.
        """
        selected_data = pd.concat([data[ticker].loc[start_date:end_date] for ticker in tickers.keys() if ticker in data], axis=1)
        returns = selected_data.pct_change(fill_method=None)
        weighted_returns = returns.apply(lambda col: col * tickers.get(col.name, 0.0))
        mean_weighted_returns_by_date = weighted_returns.dropna().sum(axis=1)
        return mean_weighted_returns_by_date.sort_index()

    @staticmethod
    def get_data_from_pickle(file_name: str, folder_subpath: str = None):
        """
        Charge des données à partir d'un fichier pickle.

        Paramètres:
        - file_name (str): Nom du fichier pickle.
        - folder_subpath (str, optionnel): Sous-chemin du dossier contenant le fichier pickle.

        Retourne:
        - dict: Les données chargées depuis le fichier pickle.
        """
        if folder_subpath is None:
            file_path = os.path.join(os.path.dirname(__file__).replace("src\\utils", "data"), file_name + ".pkl")
        else:
            file_path = os.path.join(os.path.dirname(__file__).replace("src\\utils", "data\\"), folder_subpath + "\\", file_name + ".pkl")
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data

    @staticmethod
    def save_data_to_pickle(data, file_name, folder_subpath: str = None):
        """
        Sauvegarde des données dans un fichier pickle.

        Paramètres:
        - data (dict): Les données à sauvegarder.
        - file_name (str): Nom du fichier pickle.
        - folder_subpath (str, optionnel): Sous-chemin du dossier où sauvegarder le fichier pickle.
        """
        if folder_subpath is None:
            file_path = os.path.join(os.path.dirname(__file__).replace("src\\utils", "data"), file_name + ".pkl")
        else:
            file_path = os.path.join(os.path.dirname(__file__).replace("src\\utils", "data\\"), folder_subpath + "\\", file_name + ".pkl")
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load_asset_indices(names_list: list[str], folder_subpath: str):
        """
        Charge les stratégies à partir de fichiers pickle.

        Paramètres:
        - names_list (list[str]): Liste des noms des fichiers pickle.
        - folder_subpath (str): Sous-chemin du dossier contenant les fichiers pickle.

        Retourne:
        - dict: Dictionnaire des stratégies chargés.
        """
        asset_indices = {}
        for name in names_list:
            asset_indices.update({name: Utilities.get_data_from_pickle(name, folder_subpath)})
        return asset_indices

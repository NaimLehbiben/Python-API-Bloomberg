from datetime import datetime
import pandas as pd
from src.utils.utilities import Utilities
from src.strategies.strategies import Strategy
from src.base.quote import Quote
from src.base.position import Position
import os
from tqdm import tqdm
from src.utils import constant

class AssetIndex:
    def __init__(self, launch_date: datetime, currency: str, strategy: Strategy):
        """
        Initialisation de la classe AssetIndex.
        
        Args:
            launch_date (datetime): La date de lancement de l'indice.
            currency (str): La devise utilisée pour l'indice.
            strategy (Strategy): La stratégie utilisée pour générer des signaux de trading.
        """
        self.launch_date = launch_date
        self.currency = currency
        self.strategy = strategy
        self.last_price: Quote = None
        self.price_history: list[Quote] = []
        self.current_positions: list[Position] = []
        self.historical_position: dict[datetime, list[Position]] = {}

    def rebalance_portfolio(self, date: datetime, end_date: datetime, global_market_data: dict[str, pd.DataFrame] = None, 
                            universe: dict[str, list[str]] = None) -> None:
        """
        Rééquilibre le portefeuille selon la stratégie définie.

        Args:
            date (datetime): La date actuelle de rééquilibrage.
            end_date (datetime): La date de fin de la période de rééquilibrage.
            global_market_data (dict[str, pd.DataFrame], optional): Données de marché globales. Defaults to None.
            universe (dict[str, list[str]], optional): Univers d'investissement. Defaults to None.

        Returns:
            None
        """
        # Génère les signaux de trading et obtient le prochain date de rééquilibrage
        universe, next_date = self.strategy.generate_signals(global_market_data, universe, date, end_date)
        
        # Met à jour l'historique des prix avec les nouvelles données
        self.update_historical_prices(universe, global_market_data, date, next_date)
        
        # Met à jour le dernier prix
        self.last_price = self.price_history[-1]
        
        # Met à jour les positions actuelles et historiques
        self.update_current_and_historical_positions(date, universe)

    def update_price_history_from_list(self, new_quotes: list[Quote]) -> None:
        """
        Met à jour l'historique des prix à partir d'une liste de nouvelles cotations.

        Args:
            new_quotes (list[Quote]): Liste de nouvelles cotations.

        Returns:
            None
        """
        # Récupère les dates existantes dans l'historique des prix
        existing_dates = [quote.date for quote in self.price_history]
        
        # Ajoute les nouvelles cotations si la date n'existe pas déjà
        for new_quote in new_quotes:
            if new_quote.date not in existing_dates:
                self.price_history.append(new_quote)
                existing_dates.append(new_quote.date)
        
        # Trie l'historique des prix par date
        self.price_history = sorted(self.price_history, key=lambda x: x.date)

    def quotes_to_dataframe(self) -> pd.DataFrame:
        """
        Convertit l'historique des prix en DataFrame.

        Returns:
            pd.DataFrame: DataFrame des prix avec les dates comme index.
        """
        if not self.price_history:
            return None
        # Récupère les dates et les prix de l'historique des prix
        dates = [quote.date for quote in self.price_history]
        prices = [quote.price for quote in self.price_history]
        
        # Crée un DataFrame avec les prix et les dates comme index
        df = pd.DataFrame({'Price': prices}, index=dates)
        df.index.name = 'Date'
        return df

    def get_quote_history_from_df(self, data: pd.DataFrame) -> list[Quote]:
        """
        Convertit un DataFrame en liste de cotations.

        Args:
            data (pd.DataFrame): DataFrame des prix avec les dates comme index.

        Returns:
            list[Quote]: Liste de cotations.
        """
        # Récupère les dates et les prix du DataFrame
        dates = data.index.tolist()
        prices = data.iloc[:, 0].tolist()
        
        # Crée une liste de cotations à partir des dates et des prix
        return [Quote(date, price) for date, price in zip(dates, prices)]

    def get_last_track(self) -> float:
        """
        Obtient le dernier prix de l'historique des prix.

        Returns:
            float: Le dernier prix ou 100 si l'historique est vide.
        """
        if not self.price_history:
            return 100
        else:
            # Trie l'historique des prix par date
            self.price_history = sorted(self.price_history, key=lambda x: x.date)
            return self.price_history[-1].price

    def update_historical_prices(self, new_weights: dict[str, float], market_data: dict[str, pd.DataFrame], date: datetime, next_date: datetime) -> None:
        """
        Met à jour l'historique des prix avec de nouvelles pondérations et données de marché.

        Args:
            new_weights (dict[str, float]): Nouvelles pondérations des actifs.
            market_data (dict[str, pd.DataFrame]): Données de marché.
            date (datetime): Date actuelle.
            next_date (datetime): Prochaine date de rééquilibrage.

        Returns:
            None
        """
        price_history_df = pd.DataFrame()
        return_history_df = pd.DataFrame()
        last_track = self.get_last_track()
        weighted_returns = {}
        
        # Calcule les rendements pondérés pour chaque actif
        for ticker in new_weights.keys():
            date_index = market_data[ticker].index.get_loc(date) - 1
            next_date_index = market_data[ticker].index.get_loc(next_date)
            if date == next_date:
                next_date_index += 1
            asset_price_history = market_data[ticker].iloc[date_index:next_date_index]
            weighted_returns[ticker] = asset_price_history.iloc[:, 0].pct_change().dropna() * new_weights[ticker]
        
        price_history_df = pd.DataFrame(weighted_returns)
        return_history_df["Rend"] = price_history_df.iloc[:, 1:].sum(axis=1)

        # Soustrait les coûts de transactions liés au rééquilibrage
        return_history_df["Rend"].iloc[0] -= constant.TRANSACTION_COST / 10000
        
        if not self.price_history:
            return_history_df.iloc[0, 0] = 0
        
        # Met à jour le DataFrame des prix avec les rendements cumulés
        price_history_df = pd.DataFrame({"Price": (1 + return_history_df['Rend']).cumprod() * last_track})
        new_quotes = self.get_quote_history_from_df(price_history_df)
        
        # Met à jour l'historique des prix avec les nouvelles cotations
        self.update_price_history_from_list(new_quotes)

    def update_current_and_historical_positions(self, date: datetime, weights: dict[str, float]) -> None:
        """
        Met à jour les positions actuelles et historiques du portefeuille.

        Args:
            date (datetime): Date actuelle.
            weights (dict[str, float]): Pondérations des actifs.

        Returns:
            None
        """
        # Crée des positions à partir des pondérations
        positions = [Position(ticker, weights[ticker]) for ticker in weights.keys()]
        self.current_positions = positions
        
        # Met à jour les positions historiques
        if date not in self.historical_position:
            self.historical_position[date] = positions

    def get_port_file(self, ptf_name: str) -> None:
        """
        Génère un fichier CSV contenant les informations du portefeuille.

        Args:
            ptf_name (str): Nom du portefeuille.

        Returns:
            None
        """
        data = []
        
        # Parcourt les positions historiques pour générer les données du fichier CSV
        for date, positions in self.historical_position.items():
            for position in positions:
                data.append({
                    "Portfolio Name": ptf_name.upper(),
                    "Ticker": position.ticker,
                    "Weight": position.weight,
                    "Date": date
                })
        
        # Sauvegarde les données dans un fichier CSV
        pd.DataFrame(data).to_csv(os.path.dirname(__file__).replace("src\\backtester", "data") + "\\port.csv", sep=';', index=False, decimal='.')

class BackTesting:
    @staticmethod
    def start(params: dict) -> AssetIndex:
        """
        Démarre le backtesting avec les paramètres donnés.

        Args:
            params (dict): Dictionnaire de paramètres pour le backtesting.

        Returns:
            AssetIndex: Instance de l'indice d'actifs après backtesting.
        """
        # Extraction des paramètres de backtesting
        start_date = params.get("start_date", None)
        end_date = params.get("end_date", None)
        currency = params.get("currency", None)
        ticker = params.get("ticker", None)
        use_pickle_universe = params.get("use_pickle_universe", None)
        strategy = params.get("strategy", None)
        rebalancing_frequency = params.get("rebalancing_frequency", None)
        rebalancing_moment = params.get("rebalancing_moment", None)
        risk_free_rate_ticker = params.get("risk_free_rate_ticker", None)
        weights_type = params.get("weights_type", None)
        sign = 1

        if not use_pickle_universe:
            from src.data.data_manager import DataManager

        # Création du calendrier de rééquilibrage
        rebalancing_calendar = Utilities.create_rebalancing_calendar(start_date, end_date, rebalancing_frequency, rebalancing_moment)
        if not use_pickle_universe:
            print("*** BL API scrapping is running ***")
            compositions, global_market_data = DataManager.fetch_backtest_data(start_date, end_date, ticker, currency, rebalancing_frequency, rebalancing_moment, sign)
            Utilities.save_data_to_pickle(compositions, file_name="composition", folder_subpath="universe")
            Utilities.save_data_to_pickle(global_market_data, file_name="global_market_data", folder_subpath="universe")
            other_US_data = DataManager.fetch_other_US_data(start_date, end_date, ticker, risk_free_rate_ticker,currency)
            Utilities.save_data_to_pickle(other_US_data, file_name="other_US_data", folder_subpath="universe")
        else:
            compositions = Utilities.get_data_from_pickle("composition", folder_subpath="universe")
            global_market_data = Utilities.get_data_from_pickle("global_market_data", folder_subpath="universe")
        
        # Initialisation de l'indice d'actifs
        tracker = AssetIndex(rebalancing_calendar[0], currency, strategy)
        end_date = rebalancing_calendar[-1]
        
        # Rééquilibrage du portefeuille pour chaque date dans le calendrier de rééquilibrage
        for date in tqdm(rebalancing_calendar):
            tracker.rebalance_portfolio(date, end_date, global_market_data, compositions[date])
        
        return tracker

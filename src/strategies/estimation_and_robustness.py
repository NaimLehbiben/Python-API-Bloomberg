
from scipy.stats import ttest_1samp
import os
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac
import numpy as np
from src.utils.utilities import Utilities
import src.utils.constant as constant
from sklearn.covariance import ShrunkCovariance
from scipy.optimize import minimize

class Estimation:
    
    @staticmethod
    def is_slope_positive_or_negative(slope, alpha, pos_or_neg):
        """
        Vérifie si la pente est positive ou négative en effectuant un test de Student unilatéral.

        Args:
            slope (array-like): TimeSeries de la pente.
            alpha (float): Niveau de signification pour le test.
            pos_or_neg (str): indique si tester si la pente et positive ou négative

        Returns:
            bool: True si la pente est significativement positive ou négative, False sinon.
        """
        if pos_or_neg == 'pos':
            alternative = 'greater'
        elif pos_or_neg == 'neg':
            alternative = 'less'
            
        t_statistic, p_value = ttest_1samp(slope, 0, alternative=alternative, nan_policy ="omit")
        
        # Vérification de la p-value par rapport au niveau de significativité
        return p_value < alpha
            
    

    @staticmethod
    def Cpam_FF_regress_statics(asset_indices):
        """
        Effectue une régression sur les facteurs CPAM et Fama-French des différentes stratégies.

        Args:
            asset_indices (dict): Dictionnaire contenant les indices d'actifs à analyser.

        Returns:
            dict: Un dictionnaire contenant les statistiques de régression pour chaque stratégie :
                - 'CPAM': Résultats (alpha, betas, R2) de la régression sur le facteur marché.
                - 'Fama-French': Résultats (alpha, betas, R2) de la régression sur les trois facteurs (marché, SMB, HML).
        """
        file_path = os.path.join(os.path.dirname(__file__).replace("src\\strategies", "data"), "FF_Factors.csv")
        factors_df = pd.read_csv(file_path, sep=';')/100

        statistics = {}

        for strat_name in asset_indices.keys():
            statistics[strat_name] = Estimation.__perform_factor_regression(asset_indices[strat_name], factors_df)


        return statistics
    
    @staticmethod
    def __perform_factor_regression(asset_index, factors):
        """
        Réalise la régression des rendements mensuels de la stratétgie passée en argument 
        sur les facteurs CPAM et Fama-French.

        Args:
            asset_index (object): Stratégie pour lequelle effectuer la régression.
            factors (DataFrame): DataFrame des facteurs CPAM et Fama-French.

        Returns:
            dict: Un dictionnaire contenant les résultats des régressions :
                - 'CPAM': Coefficients, t-values, R² et indicatrice de significativité pour le modèle CPAM.
                - 'Fama-French': Coefficients, t-values, R² et indicatrice de significativité pour le modèle Fama-French.
        """
        prices = asset_index.quotes_to_dataframe()
        prices.index = pd.to_datetime(prices.index)
        monthly_returns = prices.resample('M').last().pct_change().dropna()

        factors.index = monthly_returns.index
        # Initialiser un dictionnaire pour stocker les résultats
        results = {'CPAM': {}, 'Fama-French': {}}

        # Régression 1: Rendements mensuels sur le facteur marché
        mkt_factor = factors['Mkt']
        results['CPAM']['coefficients'], results['CPAM']['tvalues'], results['CPAM']['rsquared'],results['CPAM']['is_significant'] = Estimation.__regress(mkt_factor, monthly_returns)

        # Régression 2: Rendements mensuels sur les trois facteurs (marché, SMB, HML)
        three_factors = factors[['Mkt', 'SMB', 'HML']]
        results['Fama-French']['coefficients'], results['Fama-French']['tvalues'], results['Fama-French']['rsquared'],results['Fama-French']['is_significant'] = Estimation.__regress(three_factors, monthly_returns)

        return results


    @staticmethod
    def __regress(X, y):
        """
        Effectue une régression linéaire ordinaire (OLS) et calcule les t-values ajustées.

        Args:
            X (DataFrame): Variables indépendantes pour la régression.
            y (Series): Variable dépendante pour la régression.

        Returns:
            tuple: Un tuple contenant les résultats de la régression :
                - Coefficients des variables indépendantes.
                - t-values ajustées des coefficients.
                - R² du modèle.
                - Indicatrices de significativité des coefficients ('*', '**', '').
        """
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        nw_cov = cov_hac(model)
        nw_tvalues = model.params / np.sqrt(np.diag(nw_cov))
        significance_indic = [
            '**' if np.abs(t) > 1.96 else '*' if np.abs(t) > 1.645 else '' for t in nw_tvalues
        ]

        return model.params, nw_tvalues, model.rsquared, significance_indic

    @staticmethod
    def __calc_cov_matrix(decile, market_data, date, frequency, rebalance_at):
        """
        Calcule la matrice de covariance ajustée pour un décile spécifique à une date donnée.

        Args:
            decile (list): Liste des tickers constituant le décile.
            market_data (dict): Dictionnaire des données de marché.
            date (str): Date pour laquelle calculer la matrice de covariance.
            frequency (str): Fréquence des données 
            rebalance_at (str): Moment de rebalancement

        Returns:
            DataFrame: Matrice de covariance ajustée pour le décile spécifié à la date donnée.
        """
        previous_date = Utilities.get_rebalancing_date(date, sign = -1, frequency=frequency, 
                                                       rebalance_at=rebalance_at,step = constant.STEP_VOL)
        returns_df  = pd.concat([market_data[ticker].loc[previous_date:date] 
                                 for ticker in decile 
                                 if ticker in market_data], axis=1).pct_change().dropna()
        cov_matrix = ShrunkCovariance().fit(returns_df).covariance_
        return cov_matrix
    
    @staticmethod
    def __calc_diversification_ratio(weights, cov_matrix):
        """
        Calcule le ratio de diversification pour un portefeuille avec des poids donnés 
        et une matrice de covariance.

        Args:
            weights (array-like): Poids des actifs dans le portefeuille.
            cov_matrix (DataFrame): Matrice de covariance des rendements des actifs.

        Returns:
            float: Ratio de diversification calculé.
        """
        weighted_vol = np.sqrt(np.diag(cov_matrix) @ weights.T)
        ptf_vol = np.sqrt(weights.T @ cov_matrix @ weights)
        diversification_ratio = weighted_vol / ptf_vol

        return -diversification_ratio

    @staticmethod
    def optimize_diversification_ratio(decile, market_data, date, weights, frequency, rebalancing_at):
        """
        Optimise les poids du portefeuille pour maximiser le ratio de diversification.

        Args:
            decile (list): Liste des tickers constituant le décile à optimiser.
            market_data (dict): Dictionnaire des données de marché.
            date (str): Date à laquelle optimiser les poids.
            weights (dict): Dictionnaire des poids initiaux des actifs dans le portefeuille.
            frequency (str): Fréquence des données 
            rebalancing_at (str): Moment de rebalancement

        Returns:
            dict: Dictionnaire des poids optimisés pour chaque ticker dans le décile.
        """
        cov = Estimation.__calc_cov_matrix(decile, market_data, date, frequency, rebalancing_at)
        optimal_weights = minimize(Estimation.__calc_diversification_ratio, x0=np.array(list(weights.values())), args=cov,
                                        method='SLSQP', bounds = tuple((0.01, 1) for w in weights.values()), 
                                        constraints=({'type': 'eq', 'fun': lambda x: 1 - np.sum(x)}))
        return {ticker : weight for ticker, weight in zip(decile, optimal_weights['x'])}



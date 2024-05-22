
from scipy.stats import ttest_1samp
import os
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac
import numpy as np

class Estimation:
    
    @staticmethod
    def is_slope_positive_or_negative(slope, alpha, pos_or_neg):
        
        if pos_or_neg == 'pos':
            alternative = 'greater'
        elif pos_or_neg == 'neg':
            alternative = 'less'
            
        # Test de Student t à une seule échantillon 
        t_statistic, p_value = ttest_1samp(slope, 0, alternative=alternative, nan_policy ="omit")
        
        # Vérification de la p-value par rapport au niveau de significativité
        return p_value < alpha
    

    @staticmethod
    def Cpam_FF_regress_statics(asset_indices):
        
        file_path = os.path.join(os.path.dirname(__file__).replace("src\\strategies", "data"), "FF_Factors.csv")
        factors_df = pd.read_csv(file_path, sep=';')/100

        statistics = {}

        for strat_name in asset_indices.keys():
            statistics[strat_name] = Estimation.__perform_factor_regression(asset_indices[strat_name], factors_df)


        return statistics
    
    @staticmethod
    def __perform_factor_regression(asset_index, factors):

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
        X = sm.add_constant(X)
        model = sm.OLS(y, X).fit()
        nw_cov = cov_hac(model)
        nw_tvalues = model.params / np.sqrt(np.diag(nw_cov))
        significance_indic = [
            '**' if np.abs(t) > 1.96 else '*' if np.abs(t) > 1.645 else '' for t in nw_tvalues
        ]

        return model.params, nw_tvalues, model.rsquared, significance_indic




from scipy.stats import ttest_1samp

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
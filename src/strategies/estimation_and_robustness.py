
from scipy.stats import ttest_1samp

class Estimation:
    
    @staticmethod
    def is_slope_significantly_positive(slope, alpha):
        # Test de Student t à une seule échantillon avec alternative='greater'
        t_statistic, p_value = ttest_1samp(slope, 0, alternative='greater', nan_policy ="omit")
        
        # Vérification de la p-value par rapport au niveau de significativité
        return p_value < alpha
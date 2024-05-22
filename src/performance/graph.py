import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from performance.metrics import MetricsCalculator
from strategies.estimation_and_robustness import Estimation
import pandas as pd
import numpy as np

class IndexPlotter:
    """
    Class for computing the performance and risk of a asset index.
    """

    @staticmethod
    def plot_track_records(asset_indices, nber_data):
        """
        Plot the track record of the index.

        Parameters:
        - index: The financial index for plotting.
        """

        # Création du graphique
        fig, ax = plt.subplots()
        
        
        if nber_data is not None:
            nber_data['group'] = (nber_data['USRINDEX Index'] != nber_data['USRINDEX Index'].shift(1)).cumsum()
            [ax.axvspan(group.index[0], group.index[-1], color='dodgerblue', alpha=0.1) 
             for label, group in nber_data.groupby('group') if group['USRINDEX Index'].iloc[0] == 1]

    
        colormap = plt.get_cmap('bone')
        linestyle = ["solid", "dashdot", "solid", "dashed", "dashed"]
        
        for i, (name, asset_index) in enumerate(asset_indices.items()):
            track_df = asset_index.quotes_to_dataframe()
            ax.plot(track_df.index, track_df.iloc[:, 0], label=name, color= colormap(i / len(asset_indices)), 
                    linestyle=linestyle[i], linewidth = 0.75)
        
        if "VolatilityTiming" in asset_indices and 'VolatilityTiming2sided' not in  asset_indices:
            
            handles, labels, ax_holding = IndexPlotter._plot_holding_moments(asset_indices["VolatilityTiming"], ax)
        elif not "VolatilityTiming" in asset_indices and 'VolatilityTiming2sided' in  asset_indices:

            handles, labels, ax_holding = IndexPlotter._plot_holding_moments(asset_indices["VolatilityTiming2sided"], ax, True)
        else:
            handles, labels = ax.get_legend_handles_labels()
            ax_holding = ax

        
        # Titre et étiquettes d'axe
        ax.set_title('Performance and wealth plots of the volatility timing strategy')
        ax.set_xlabel('Date')
        ax.set_ylabel('Track Record')
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right() 
        
        handles.append(mpatches.Patch(color='dodgerblue', alpha=0.1))
        labels.append('NBER Recessions')
 
        # Affichage de la légende
        ax_holding.legend(handles=handles, labels=labels, loc='upper left', framealpha = 1)

        # Affichage du graphique
        plt.show()
    
    
    @staticmethod
    def _plot_holding_moments(asset_index, ax, is_2sided = False):

        ax_holding = ax.twinx()
        holding_high_ptf_dates = [date for date, holding in asset_index.strategy.switch.items() if holding == 'High']
        
        for date in holding_high_ptf_dates:
            ax_holding.axvline(x=date, color='grey', linestyle='-', linewidth=0.2)
            
        
        handles, labels = ax.get_legend_handles_labels()
        handles.append(plt.Line2D([], [], color='grey', linewidth=0.2))
        labels.append('Holding High Portofolio')
        
 
        if is_2sided:
            
            holding_low_ptf_dates = [date for date, holding in asset_index.strategy.switch.items() if holding == 'Low']
            
            for date in holding_low_ptf_dates:
                ax_holding.axvline(x=date, color='lightseagreen', linestyle='-', linewidth=0.2)
                
            handles.append(plt.Line2D([], [], color='lightseagreen', linewidth=0.2))
            labels.append('Holding Low Portofolio')
        
        ax_holding.yaxis.set_visible(False)
        return handles, labels, ax_holding
    
    @staticmethod
    def plot_tracks_for_diff_rebalncing_freq(asset_indices, label_names, strat_name):

        # Création du graphique
        fig, ax = plt.subplots()

        colormap = plt.get_cmap('bone')
        linestyle = ["solid", "dashdot", "solid", "dashed"]
        
        for i, asset_index in enumerate(asset_indices):
            track_df = asset_index.quotes_to_dataframe()
            ax.plot(track_df.index, track_df.iloc[:, 0], label=label_names[i], color= colormap(i / len(asset_indices)), 
                    linestyle=linestyle[i], linewidth = 0.75)

        # Titre et étiquettes d'axe
        ax.set_title(f'{strat_name} - Wealth plot under different rebalncing frequencies')
        ax.set_xlabel('Date')
        ax.set_ylabel('Track Record')
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right() 
        
        # Affichage de la légende
        ax.legend(loc='upper left', framealpha = 1)

        # Affichage du graphique
        plt.show()


    @staticmethod
    def asset_indices_barplot(asset_indices, other_data):

        metrics_calculator = MetricsCalculator(other_data)
        strat_names = ["HighVolatilityDecile", "VolatilityTiming2sided", "MidVolatilityDecile","VolatilityTiming","LowVolatilityDecile"]
        annualized_returns = [metrics_calculator.calculate_return(asset_indices[name]) for name in strat_names]
        annualized_vol = [metrics_calculator.calculate_volatility(asset_index) for asset_index in asset_indices.values()]
        annualized_sharpe = [metrics_calculator.calculate_sharpe_ratio(asset_indices[name]) for name in strat_names]


        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        # Premier graphique: Bar plot des volatilités
        ax[0].bar(['Low','Mid','High','Vol \nTiming','Vol \n2sided'], annualized_vol, color='blue', width=0.8)
        ax[0].set_ylabel('Volatilty')


        # Deuxième graphique: Bar plot des returns + plot des ratios de Sharpe
        ax2 = ax[1].twinx()  # Deuxième axe pour le ratio de Sharpe
        ax[1].bar(['High','Vol \n2sided', 'Mid','Vol \nTiming','Low'], annualized_returns, color='green', width=0.8, align='center')
        ax[1].set_ylabel('Return')

        ax2.plot(['High','Vol \n2sided', 'Mid','Vol \nTiming','Low'], annualized_sharpe, color='black', marker='o', linestyle='-', linewidth=2, markersize=8, label ='Sharpe')
        ax2.set_ylabel('Sharpe')
        ax2.legend(loc='upper left')

        fig.tight_layout()
        plt.show()
    

    @staticmethod
    def asset_indices_plot_under_diff_conditions(asset_indices, other_data):

        metrics_calculator = MetricsCalculator(other_data)
        avg_good_mkt, avg_bad_mkt = metrics_calculator._calc_good_bad_mkt_stats(asset_indices)

        strat_names = ["LowVolatilityDecile","MidVolatilityDecile","HighVolatilityDecile"]
        good_mkt_volatility = [avg_good_mkt[name][1] for name in strat_names]
        bad_mkt_volatility = [avg_bad_mkt[name][1] for name in strat_names]
        good_mkt_returns = [avg_good_mkt[name][0] for name in strat_names]
        bad_mkt_returns = [avg_bad_mkt[name][0] for name in strat_names]


        # Définir la largeur des barres
        width = 0.35

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        # Positions des barres
        x = range(len(strat_names))

        # Premier graphique: Bar plot des volatilités selon les conditions de marché
        ax[0].bar(x, good_mkt_volatility, width, label='Good Market', color='blue', align='center')
        ax[0].bar([i + width for i in x], bad_mkt_volatility, width, label='Bad Market', color='red', align='center')
        ax[0].set_ylabel('Volatility')
        ax[0].set_xticks([i + width / 2 for i in x])
        ax[0].set_xticklabels(['Low','Mid','High'])


        # Deuxième graphique: Bar plot des returns selon les conditions de marché
        ax[1].bar(x, good_mkt_returns, width, color='blue', align='center')
        ax[1].bar([i + width for i in x], bad_mkt_returns, width, color='red', align='center')
        ax[1].set_ylabel('Return')
        ax[1].set_xticks([i + width / 2 for i in x])
        ax[1].set_xticklabels(['Low','Mid','High'])

        # Ajouter une légende globale pour la figure entière
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)

        fig.tight_layout()
        plt.show()
    
    @staticmethod
    def display_asset_indices_metrics(asset_indices, other_data):
        
        # Calcul des métriques de performance
        metrics_calculator = MetricsCalculator(other_data)

        # Charger les données des portefeuilles
        low_vol_data = asset_indices['LowVolatilityDecile']
        mid_vol_data = asset_indices['MidVolatilityDecile']
        high_vol_data = asset_indices['HighVolatilityDecile']
        vol_timing_data = asset_indices['VolatilityTiming']
        vol_timing_2sided_data = asset_indices['VolatilityTiming2sided']

        # Calculer les métriques pour chaque portefeuille
        low_vol_metrics = metrics_calculator.calculate_all_metrics(low_vol_data, low_vol_data)
        mid_vol_metrics = metrics_calculator.calculate_all_metrics(mid_vol_data, low_vol_data)
        high_vol_metrics = metrics_calculator.calculate_all_metrics(high_vol_data, low_vol_data)
        vol_timing_metrics = metrics_calculator.calculate_all_metrics(vol_timing_data, low_vol_data)
        vol_timing_2sided_metrics = metrics_calculator.calculate_all_metrics(vol_timing_2sided_data, low_vol_data)

        # Compilation des métriques dans un DataFrame
        metrics_df = pd.DataFrame({
            'Low Volatility': low_vol_metrics,
            'Mid Volatility': mid_vol_metrics,
            'High Volatility': high_vol_metrics,
            'Volatility Timing': vol_timing_metrics,
            'Volatility Timing 2-sided': vol_timing_2sided_metrics
        })


        metrics_df = metrics_df.applymap(lambda x: x.item() if isinstance(x, pd.Series) else x)
        metrics_df.update(metrics_df.loc[['Return', 'Volatility', 'Max Drawdown', 'SQRT (Semi-variance)']].applymap(lambda x: f"{x:.2f}%" if pd.notnull(x) else "NaN"))


        metrics_df.replace([-np.inf, np.inf], 'Benchmark', inplace=True)

        return metrics_df

    @staticmethod
    def display_regress_statistics(asset_indices):

        stats = Estimation.Cpam_FF_regress_statics(asset_indices)

        rows_CPAM = []
        rows_FF = []
        ["LowVolatilityDecile", "MidVolatilityDecile", "HighVolatilityDecile", "VolatilityTiming", "VolatilityTiming2sided"]
        strat_short_names = {'LowVolatilityDecile' : 'LowVol',
                             'MidVolatilityDecile' : 'MidVol',
                             'HighVolatilityDecile' : 'HighVol',
                             'VolatilityTiming' : 'VolTiming',
                             'VolatilityTiming2sided' : 'Vol2Sided'}
        
        columns_names_CPAM = ['α (%)', 'β_mkt', 'R²']
        columns_names_FF = ['α (%)', 'β_mkt',  'β_SMB', 'β_HML', 'R²']
        index_names = [item for strat_name in asset_indices.keys() for item in (strat_short_names[strat_name], '')]


        for _, models in stats.items():
            for model_name, metrics in models.items():
                coefficients = metrics['coefficients']
                tvalues = metrics['tvalues']
                rsquared = metrics['rsquared']
                is_significant = metrics['is_significant']
                
                if model_name == 'CPAM':
                    rows_CPAM.append([ f"{coefficients['const']*100:.3}{is_significant[0]}", 
                         f"{coefficients['Mkt']:.3f}{is_significant[1]}",f"{rsquared:.3f}"])
                    rows_CPAM.append([f"({tvalues['const']:.3f})",f"({tvalues['Mkt']:.3f})",""])
                else:
                    rows_FF.append([f"{coefficients['const']*100:.3}{is_significant[0]}",  
                        f"{coefficients['Mkt']:.3f}{is_significant[1]}",  
                        f"{coefficients['SMB']:.3f}{is_significant[2]}", 
                        f"{coefficients['HML']:.3f}{is_significant[3]}", f"{rsquared:.3f}"])
                    rows_FF.append([f"({tvalues['const']:.3f})",f"({tvalues['Mkt']:.3f})",
                                    f"({tvalues['SMB']:.3f})", f"({tvalues['HML']:.3f})", ""])
        
        return pd.DataFrame(rows_CPAM, columns=columns_names_CPAM, index= index_names), pd.DataFrame(rows_FF, columns=columns_names_FF, index=index_names)
    






    

            
        

        
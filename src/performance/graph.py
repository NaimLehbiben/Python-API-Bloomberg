import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from performance.metrics import MetricsCalculator
from strategies.estimation_and_robustness import Estimation
import pandas as pd
import numpy as np

class IndexPlotter:

    @staticmethod
    def plot_track_records(asset_indices, nber_data):
        """
        Trace les track de plusieurs stratégies

        Paramètres :
        - asset_indices : Dictionnaire des stratégies.
        - nber_data : Données de récession NBER.
        """
        fig, ax = plt.subplots()
        
        # Affichage de l'indicatrice NBER
        if nber_data is not None:
            nber_data['group'] = (nber_data['USRINDEX Index'] != nber_data['USRINDEX Index'].shift(1)).cumsum()
            [ax.axvspan(group.index[0], group.index[-1], color='dodgerblue', alpha=0.1) 
             for label, group in nber_data.groupby('group') if group['USRINDEX Index'].iloc[0] == 1]

        colormap = plt.get_cmap('Set1')
        linestyle = ["solid", "dashdot", "solid", "dashed", "dashed"]
        
        # Affichage des track des différentes stratégies
        for i, (name, asset_index) in enumerate(asset_indices.items()):
            track_df = asset_index.quotes_to_dataframe()
            ax.plot(track_df.index, track_df.iloc[:, 0], label=name, color=colormap.colors[i % len(colormap.colors)],
                    linestyle=linestyle[i], linewidth=0.5)
        
        # Affichages des moments de détention des portefeuilles constituant les stratégies
        if "VolatilityTiming" in asset_indices and 'VolatilityTiming2sided' not in asset_indices:
            handles, labels, ax_holding = IndexPlotter._plot_holding_moments(asset_indices["VolatilityTiming"], ax)
        elif "VolatilityTiming2sided" in asset_indices:
            handles, labels, ax_holding = IndexPlotter._plot_holding_moments(asset_indices["VolatilityTiming2sided"], ax, True)
        else:
            handles, labels = ax.get_legend_handles_labels()
            ax_holding = ax

        ax.set_title('Performance and wealth plots of the volatility timing strategy')
        ax.set_xlabel('Date')
        ax.set_ylabel('Track Record')
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right() 
        
        handles.append(mpatches.Patch(color='dodgerblue', alpha=0.1))
        labels.append('NBER Recessions')
 
        ax_holding.legend(handles=handles, labels=labels, loc='upper left', framealpha=1)

        plt.show()
    
    @staticmethod
    def _plot_holding_moments(asset_index, ax, is_2sided=False):
        """
        Trace les moments de conservation des portefuilles alternatifs composant les 
        stratégies de timing de volatilité

        Paramètres :
        - asset_index : La stratggie pour lequellle tracer les moments de conservation.
        - ax : L'axe sur lequel tracer.
        - is_2sided : Indique si la startégie de timing de volatilité est la 2sided ou la version base.
        """
        ax_holding = ax.twinx()
        holding_high_ptf_dates = [date for date, holding in asset_index.strategy.ptf_hold.items() if holding == 'High']
        
        for date in holding_high_ptf_dates:
            ax_holding.axvline(x=date, color='grey', linestyle='-', linewidth=0.2)
        
        handles, labels = ax.get_legend_handles_labels()
        handles.append(plt.Line2D([], [], color='grey', linewidth=0.2))
        labels.append('Holding High Portfolio')
        
        if is_2sided:
            holding_low_ptf_dates = [date for date, holding in asset_index.strategy.ptf_hold.items() if holding == 'Low']
            for date in holding_low_ptf_dates:
                ax_holding.axvline(x=date, color='lightseagreen', linestyle='-', linewidth=0.2)
            handles.append(plt.Line2D([], [], color='lightseagreen', linewidth=0.2))
            labels.append('Holding Low Portfolio')
        
        ax_holding.yaxis.set_visible(False)
        return handles, labels, ax_holding
    
    @staticmethod
    def plot_tracks_general(asset_indices, label_names, graph_title):
        """
        Trace les track pour pluisuers stratégies 

        Paramètres :
        - asset_indices : Dictionnaire des stratégies.
        - label_names : Liste des étiquettes pour le tracé.
        - graph_title : Titre du graphique.
        """
        fig, ax = plt.subplots()

        colormap = plt.get_cmap('Set1')
        linestyle = ["solid", "dashdot", "solid", "dashed"]
        
        for i, asset_index in enumerate(asset_indices):
            track_df = asset_index.quotes_to_dataframe()
            ax.plot(track_df.index, track_df.iloc[:, 0], label=label_names[i], 
                    color=colormap.colors[i % len(colormap.colors)], 
                    linestyle=linestyle[i], linewidth=0.75)

        ax.set_title(graph_title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Track Record')
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right() 
        
        ax.legend(loc='upper left', framealpha=1)

        plt.show()

    @staticmethod
    def asset_indices_barplot(asset_indices, other_data, risk_free_rate_ticker):
        """
        Affiche les barplot des rendements, ratio de sharpe et volatilités des différentes stratégies

        Paramètres :
        - asset_indices : Dictionnaire des startégies.
        - other_data : Autres données pertinentes.
        - risk_free_rate_ticker : Le ticker du taux sans risque.
        """
        metrics_calculator = MetricsCalculator(other_data, risk_free_rate_ticker)
        strat_names = ["HighVolatilityDecile", "VolatilityTiming2sided", "MidVolatilityDecile","VolatilityTiming","LowVolatilityDecile"]
        annualized_returns = [metrics_calculator.calculate_annualized_return(asset_indices[name]) for name in strat_names]
        annualized_vol = [metrics_calculator.calculate_volatility(asset_index) for asset_index in asset_indices.values()]
        annualized_sharpe = [metrics_calculator.calculate_sharpe_ratio(asset_indices[name]) for name in strat_names]

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        ax[0].bar(['Low','Mid','High','Vol \nTiming','Vol \n2sided'], annualized_vol, color='blue', width=0.8)
        ax[0].set_ylabel('Volatility')

        ax2 = ax[1].twinx()
        ax[1].bar(['High','Vol \n2sided', 'Mid','Vol \nTiming','Low'], annualized_returns, color='green', width=0.8, align='center')
        ax[1].set_ylabel('Return')

        ax2.plot(['High','Vol \n2sided', 'Mid','Vol \nTiming','Low'], annualized_sharpe, color='black', marker='o', linestyle='-', linewidth=2, markersize=8, label='Sharpe')
        ax2.set_ylabel('Sharpe')
        ax2.legend(loc='upper left')

        fig.tight_layout()
        plt.show()

    @staticmethod
    def asset_indices_plot_under_diff_conditions(asset_indices, other_data, risk_free_rate_ticker, start_date, end_date, frequency, rebalance_at, ticker):
        """
        Affiche des graphiques à barres des différentes stratégies dans différentes conditions de marché.

        Paramètres :
        - asset_indices : Dictionnaire des stratégies.
        - other_data : Autres données pertinentes.
        - risk_free_rate_ticker : Le ticker du taux sans risque.
        - start_date : Date de début de l'analyse.
        - end_date : Date de fin de l'analyse.
        - frequency : Fréquence de rebalancement.
        - rebalance_at : Moment de rebalancement.
        - ticker : Ticker de l'indice.
        """
        metrics_calculator = MetricsCalculator(other_data, risk_free_rate_ticker)
        avg_good_mkt, avg_bad_mkt = metrics_calculator._calc_good_bad_mkt_stats(asset_indices, start_date, end_date, frequency, rebalance_at, ticker)

        strat_names = ["LowVolatilityDecile","MidVolatilityDecile","HighVolatilityDecile"]
        good_mkt_volatility = [avg_good_mkt[name][1] for name in strat_names]
        bad_mkt_volatility = [avg_bad_mkt[name][1] for name in strat_names]
        good_mkt_returns = [avg_good_mkt[name][0] for name in strat_names]
        bad_mkt_returns = [avg_bad_mkt[name][0] for name in strat_names]

        width = 0.35

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        x = range(len(strat_names))

        ax[0].bar(x, good_mkt_volatility, width, label='Good Market', color='blue', align='center')
        ax[0].bar([i + width for i in x], bad_mkt_volatility, width, label='Bad Market', color='red', align='center')
        ax[0].set_ylabel('Volatility')
        ax[0].set_xticks([i + width / 2 for i in x])
        ax[0].set_xticklabels(['Low','Mid','High'])

        ax[1].bar(x, good_mkt_returns, width, color='blue', align='center')
        ax[1].bar([i + width for i in x], bad_mkt_returns, width, color='red', align='center')
        ax[1].set_ylabel('Return')
        ax[1].set_xticks([i + width / 2 for i in x])
        ax[1].set_xticklabels(['Low','Mid','High'])

        fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=2)

        fig.tight_layout()
        plt.show()

    @staticmethod
    def display_asset_indices_metrics(asset_indices, other_data, risk_free_rate_ticker):
        """
        Affiche les métriques de performance et risques des stratégies.

        Paramètres :
        - asset_indices : Dictionnaire des startégies.
        - other_data : Autres données pertinentes.
        - risk_free_rate_ticker : Le ticker du taux sans risque.

        Retourne :
        - DataFrame contenant les métriques calculées pour chaque startégies.
        """
        metrics_calculator = MetricsCalculator(other_data, risk_free_rate_ticker)

        low_vol_data = asset_indices['LowVolatilityDecile']
        mid_vol_data = asset_indices['MidVolatilityDecile']
        high_vol_data = asset_indices['HighVolatilityDecile']
        vol_timing_data = asset_indices['VolatilityTiming']
        vol_timing_2sided_data = asset_indices['VolatilityTiming2sided']

        low_vol_metrics = metrics_calculator.calculate_all_metrics(low_vol_data, low_vol_data)
        mid_vol_metrics = metrics_calculator.calculate_all_metrics(mid_vol_data, low_vol_data)
        high_vol_metrics = metrics_calculator.calculate_all_metrics(high_vol_data, low_vol_data)
        vol_timing_metrics = metrics_calculator.calculate_all_metrics(vol_timing_data, low_vol_data)
        vol_timing_2sided_metrics = metrics_calculator.calculate_all_metrics(vol_timing_2sided_data, low_vol_data)

        metrics_df = pd.DataFrame({
            'Low Volatility': low_vol_metrics,
            'Mid Volatility': mid_vol_metrics,
            'High Volatility': high_vol_metrics,
            'Volatility Timing': vol_timing_metrics,
            'Volatility Timing 2-sided': vol_timing_2sided_metrics
        })

        metrics_df = metrics_df.applymap(lambda x: x.item() if isinstance(x, pd.Series) else x)
        metrics_df.update(metrics_df.loc[['Total Return','Annualized Return', 'Annualized Volatility', 'Monthly Volatility', 'Daily Volatility', 'Max Drawdown', 'SQRT (Semi-variance)', 'Historical VaR (95%)']].applymap(lambda x: f"{x:.2f}%" if pd.notnull(x) else "NaN"))

        metrics_df.replace([-np.inf, np.inf], 'Benchmark', inplace=True)

        return metrics_df

    @staticmethod
    def display_regress_statistics(asset_indices):
        """
        Affiche les statistiques de régression pour les différentes startégies.

        Paramètres :
        - asset_indices : Dictionnaire des startégies.

        Retourne :
        - DataFrame contenant les statistiques de régression CPAM et Fama-French.
        """
        stats = Estimation.Cpam_FF_regress_statics(asset_indices)

        rows_CPAM = []
        rows_FF = []
        strat_short_names = {'LowVolatilityDecile' : 'LowVol',
                             'MidVolatilityDecile' : 'MidVol',
                             'HighVolatilityDecile' : 'HighVol',
                             'VolatilityTiming' : 'VolTiming',
                             'VolatilityTiming2sided' : 'Vol2Sided'}
        
        columns_names_CPAM = ['α (%)', 'β_mkt', 'R²']
        columns_names_FF = ['α (%)', 'β_mkt', 'β_SMB', 'β_HML', 'R²']
        index_names = [item for strat_name in asset_indices.keys() for item in (strat_short_names[strat_name], '')]

        for _, models in stats.items():
            for model_name, metrics in models.items():
                coefficients = metrics['coefficients']
                tvalues = metrics['tvalues']
                rsquared = metrics['rsquared']
                is_significant = metrics['is_significant']
                
                if model_name == 'CPAM':
                    rows_CPAM.append([f"{coefficients['const']*100:.3}{is_significant[0]}", 
                                      f"{coefficients['Mkt']:.3f}{is_significant[1]}", f"{rsquared:.3f}"])
                    rows_CPAM.append([f"({tvalues['const']:.3f})", f"({tvalues['Mkt']:.3f})", ""])
                else:
                    rows_FF.append([f"{coefficients['const']*100:.3}{is_significant[0]}",  
                                    f"{coefficients['Mkt']:.3f}{is_significant[1]}",  
                                    f"{coefficients['SMB']:.3f}{is_significant[2]}", 
                                    f"{coefficients['HML']:.3f}{is_significant[3]}", f"{rsquared:.3f}"])
                    rows_FF.append([f"({tvalues['const']:.3f})", f"({tvalues['Mkt']:.3f})",
                                    f"({tvalues['SMB']:.3f})", f"({tvalues['HML']:.3f})", ""])
        
        return pd.DataFrame(rows_CPAM, columns=columns_names_CPAM, index=index_names), pd.DataFrame(rows_FF, columns=columns_names_FF, index=index_names)


    @staticmethod
    def display_joint_metrics(*dfs, label_names, column_names):
        """
        Affiche les métriques combinées à partir de plusieurs DataFrames.

        Paramètres :
        - *dfs : Liste de DataFrames à concaténer.
        - label_names : Noms des colonnes à sélectionner dans chaque DataFrame.
        - column_names : Noms des sous-colonnes pour le DataFrame résultant.

        Retourne :
        - DataFrame combiné avec les métriques sélectionnées et renommées.
        """
        concatenated_dfs = []

        for idx, metrics_df in enumerate(dfs):
            # Sélectionner uniquement les colonnes spécifiées dans label_names
            metrics_df_selected = metrics_df[label_names]

            # Renommer les colonnes pour inclure l'indice du DataFrame
            metrics_df_selected.columns = [f'{col}_{idx+1}' for col in metrics_df_selected.columns]

            # Ajouter le DataFrame traité à la liste
            concatenated_dfs.append(metrics_df_selected)

        # Concaténer les DataFrames en respectant l'ordre des colonnes
        combined_df = pd.concat(concatenated_dfs, axis=1)

        # Trier les colonnes dans l'ordre spécifié par label_names
        combined_df = combined_df.reindex(sorted(combined_df.columns), axis=1)

        # Renommer les colonnes avec les noms spécifiés dans column_names
        combined_df.columns = pd.MultiIndex.from_product([label_names, column_names])

        return combined_df

    @staticmethod
    def disp_switch_stats(asset_indices_monthly, other_data, risk_free_rate_ticker):
        """
        Affiche les statistiques de passage du portefuille de base aux portefuilles alternatifs
        pour les différentes stratégies.

        Paramètres :
        - asset_indices_monthly : Dictionnaire des stratégies.
        - other_data : Autres données pertinentes.
        - risk_free_rate_ticker : Le ticker du taux sans risque.

        Retourne :
        - DataFrame contenant les statistiques.
        """
        metrics_calculator = MetricsCalculator(other_data, risk_free_rate_ticker)
        switch_stats = metrics_calculator.calculate_switch_performance(asset_indices_monthly,"monthly")

        if "VolatilityTiming" in asset_indices_monthly.keys():
            total_switches_str = f"Low-/High-vol split: {round(switch_stats['Holding Low Percentage'])}%/{round(switch_stats['Holding High Percentage'])}%"

        else:
            holding_mid_percentage =  100 -(switch_stats['Holding Low Percentage'] + switch_stats['Holding High Percentage'])
            total_switches_str = f"Low-/Mid-/High-vol split: {round(switch_stats['Holding Low Percentage'])}%/{round(holding_mid_percentage)}%/{round(switch_stats['Holding High Percentage'])}%"

        df = pd.DataFrame({
            'Correct Switches': [f"{round(switch_stats['Correct Switch Percentage'], 1)}%", f"{round(switch_stats['Correct Switch Total Performance'], 1)}%"],
            'Wrong Switches': [f"{round(switch_stats['Incorrect Switch Percentage'], 1)}%", f"{round(switch_stats['Incorrect Switch Total Performance'], 1)}%"],
            'Total Switches': [total_switches_str, ""]
        }, index=['Percentage', 'Total Return Out Performance'])

        return df
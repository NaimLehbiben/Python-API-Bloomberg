import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from performance.metrics import MetricsCalculator
from strategies.estimation_and_robustness import Estimation
import pandas as pd
import numpy as np

class IndexPlotter:
    """
    Class for computing the performance and risk of an asset index.
    """

    @staticmethod
    def plot_track_records(asset_indices, nber_data):
        """
        Plot the track record of the index.

        Parameters:
        - asset_indices: Dictionary of asset indices.
        - nber_data: NBER recession data.
        """
        fig, ax = plt.subplots()
        
        if nber_data is not None:
            nber_data['group'] = (nber_data['USRINDEX Index'] != nber_data['USRINDEX Index'].shift(1)).cumsum()
            [ax.axvspan(group.index[0], group.index[-1], color='dodgerblue', alpha=0.1) 
             for label, group in nber_data.groupby('group') if group['USRINDEX Index'].iloc[0] == 1]

        colormap = plt.get_cmap('Set1')
        linestyle = ["solid", "dashdot", "solid", "dashed", "dashed"]
        
        for i, (name, asset_index) in enumerate(asset_indices.items()):
            track_df = asset_index.quotes_to_dataframe()
            ax.plot(track_df.index, track_df.iloc[:, 0], label=name, color=colormap.colors[i % len(colormap.colors)],
                    linestyle=linestyle[i], linewidth=0.5)
        
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
        Plot holding moments on the provided axis.

        Parameters:
        - asset_index: The asset index to plot holding moments for.
        - ax: The axis to plot on.
        - is_2sided: Whether to plot 2-sided holding moments.
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
        Plot the track records for different rebalancing frequencies.

        Parameters:
        - asset_indices: List of asset indices.
        - label_names: List of labels for the plot.
        - strat_name: Strategy name for the title.
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
        Plot bar charts for asset indices.

        Parameters:
        - asset_indices: Dictionary of asset indices.
        - other_data: Other relevant data.
        - risk_free_rate_ticker: The ticker for the risk-free rate.
        """
        metrics_calculator = MetricsCalculator(other_data, risk_free_rate_ticker)
        strat_names = ["HighVolatilityDecile", "VolatilityTiming2sided", "MidVolatilityDecile","VolatilityTiming","LowVolatilityDecile"]
        annualized_returns = [metrics_calculator.calculate_annualized_return(asset_indices[name]) for name in strat_names]
        annualized_vol = [metrics_calculator.calculate_volatility(asset_index, period='annual') for asset_index in asset_indices.values()]
        monthly_vol = [metrics_calculator.calculate_volatility(asset_index, period='monthly') for asset_index in asset_indices.values()]
        daily_vol = [metrics_calculator.calculate_volatility(asset_index, period='daily') for asset_index in asset_indices.values()]
        annualized_sharpe = [metrics_calculator.calculate_sharpe_ratio(asset_indices[name]) for name in strat_names]
        var_95 = [metrics_calculator.calculate_var(asset_index, confidence_level=0.95) for asset_index in asset_indices.values()]

        fig, ax = plt.subplots(2, 2, figsize=(14, 10))

        # Annualized Volatility
        ax[0, 0].bar(['Low','Mid','High','Vol \nTiming','Vol \n2sided'], annualized_vol, color='blue', width=0.8)
        ax[0, 0].set_ylabel('Annualized Volatility')

        # Monthly Volatility
        ax[0, 1].bar(['Low','Mid','High','Vol \nTiming','Vol \n2sided'], monthly_vol, color='blue', width=0.8)
        ax[0, 1].set_ylabel('Monthly Volatility')

        # Daily Volatility
        ax[1, 0].bar(['Low','Mid','High','Vol \nTiming','Vol \n2sided'], daily_vol, color='blue', width=0.8)
        ax[1, 0].set_ylabel('Daily Volatility')

        # VaR 95%
        ax[1, 1].bar(['Low','Mid','High','Vol \nTiming','Vol \n2sided'], var_95, color='blue', width=0.8)
        ax[1, 1].set_ylabel('VaR 95%')

        for a in ax.flat:
            a.set_xticklabels(['Low','Mid','High','Vol \nTiming','Vol \n2sided'], rotation=45, ha='right')

        fig.tight_layout()
        plt.show()

    @staticmethod
    def asset_indices_plot_under_diff_conditions(asset_indices, other_data, risk_free_rate_ticker, start_date, end_date, frequency, rebalance_at, ticker):
        """
        Plot bar charts for asset indices under different market conditions.

        Parameters:
        - asset_indices: Dictionary of asset indices.
        - other_data: Other relevant data.
        - risk_free_rate_ticker: The ticker for the risk-free rate.
        - start_date: The start date for the analysis.
        - end_date: The end date for the analysis.
        - frequency: Rebalancing frequency.
        - rebalance_at: Rebalancing moment.
        - ticker: Ticker for the index.
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
        Display performance metrics for asset indices.

        Parameters:
        - asset_indices: Dictionary of asset indices.
        - other_data: Other relevant data.
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
        Display regression statistics for asset indices.

        Parameters:
        - asset_indices: Dictionary of asset indices.
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

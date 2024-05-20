import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

class Analysis:
    """
    Class for computing the performance and risk of a asset index.
    """


    @staticmethod
    def plot_trackers(trackers, nber_data):
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
        
        for i, (name, tracker) in enumerate(trackers.items()):
            track_df = tracker.quotes_to_dataframe()
            ax.plot(track_df.index, track_df.iloc[:, 0], label=name, color= colormap(i / len(trackers)), 
                    linestyle=linestyle[i], linewidth = 0.75)
        
        if "VolatilityTiming" in trackers and 'VolatilityTiming2sided' not in  trackers:
            
            handles, labels, ax_holding = Analysis._plot_holding_moments(trackers["VolatilityTiming"], ax)
        elif not "VolatilityTiming" in trackers and 'VolatilityTiming2sided' in  trackers:

            handles, labels, ax_holding = Analysis._plot_holding_moments(trackers["VolatilityTiming2sided"], ax, True)
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
    def _plot_holding_moments(tracker, ax, is_2sided = False):

        ax_holding = ax.twinx()
        holding_high_ptf_dates = [date for date, holding in tracker.strategy.switch.items() if holding == 'High']
        
        for date in holding_high_ptf_dates:
            ax_holding.axvline(x=date, color='grey', linestyle='-', linewidth=0.2)
            
        
        handles, labels = ax.get_legend_handles_labels()
        handles.append(plt.Line2D([], [], color='grey', linewidth=0.2))
        labels.append('Holding High Portofolio')
        
 
        if is_2sided:
            
            holding_low_ptf_dates = [date for date, holding in tracker.strategy.switch.items() if holding == 'Low']
            
            for date in holding_low_ptf_dates:
                ax_holding.axvline(x=date, color='lightseagreen', linestyle='-', linewidth=0.2)
                
            handles.append(plt.Line2D([], [], color='lightseagreen', linewidth=0.2))
            labels.append('Holding Low Portofolio')
        
        ax_holding.yaxis.set_visible(False)
        return handles, labels, ax_holding
            
        

        
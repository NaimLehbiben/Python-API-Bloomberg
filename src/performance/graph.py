import matplotlib.pyplot as plt

class Analysis:
    """
    Class for computing the performance and risk of a asset index.
    """


    @staticmethod
    def plot(index):
        """
        Plot the track record of the index.

        Parameters:
        - index: The financial index for plotting.
        """
        track_df = index.quotes_to_dataframe()

        fig, ax = plt.subplots()

        ax.plot(track_df.index, track_df.iloc[:,0], label='Track Record', color='orange')
    
        ax.set_xlabel('Date')
        ax.set_ylabel('Index Value')

        ax.legend()
        plt.show()
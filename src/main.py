
from backtester.back_tester import BackTesting
from strategies.strategies import *
from utils.utilities import Utilities
from performance.graph import Analysis
import utils.config as config
import numpy as np

if __name__ == "__main__":
    params = {
        "currency" : config.CURRENCY,
        "start_date" : config.START_DATE,
        "end_date" : config.END_DATE,
        "ticker" : config.TICKER,
        "strategy" : VolatilityTimingStrategy(),
        "use_pickle_universe" : config.USE_PICKLE_UNIVERSE,
        }
    
    # # Lancement du back-test
    # tracker = BackTesting.start(params)

    # Graph des tracks records des strategies
    trackers = Utilities.load_trackers(["LowVolatilityDecile","MidVolatilityDecile", "HighVolatilityDecile",
                                 "VolatilityTiming", "VolatilityTiming2sided"])
    other_data = Utilities.get_data_from_pickle('other_US_data')
    Analysis.plot_trackers(trackers,other_data['USRINDEX Index'])
    
    
    # # -------------------------------------------------------------
    # # Appel de la fonction port
    # tracker = Utilities.get_data_from_pickle('HighVolatilityDecile')
    # tracker.get_port_file('VolatilityTiming')
    # # --------------------------------------------------------------

import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkcalendar import DateEntry
from datetime import datetime
import sys
import os
import warnings

# Add the parent directory to include 'src'
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))

# Imports after adjusting the sys.path
from src.backtester.back_tester import BackTesting
from src.strategies.strategies import VolatilityTimingStrategy
from src.utils.utilities import Utilities
from src.performance.graph import IndexPlotter

# Import constants
import src.utils.constant as constant

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class FinanceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Finance Backtesting Tool")

        # Ticker
        self.ticker_label = tk.Label(self, text="Ticker (e.g., RIY INDEX)")
        self.ticker_label.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)
        self.ticker_entry = tk.Entry(self)
        self.ticker_entry.grid(row=0, column=1, padx=10, pady=10)
        self.ticker_entry.insert(0, "RIY INDEX")

        # Start Date
        self.start_date_label = tk.Label(self, text="Start Date")
        self.start_date_label.grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)
        self.start_date_entry = DateEntry(self, date_pattern='yyyy-mm-dd')
        self.start_date_entry.grid(row=1, column=1, padx=10, pady=10)
        self.start_date_entry.set_date(datetime(2000, 1, 3))

        # End Date
        self.end_date_label = tk.Label(self, text="End Date")
        self.end_date_label.grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)
        self.end_date_entry = DateEntry(self, date_pattern='yyyy-mm-dd')
        self.end_date_entry.grid(row=2, column=1, padx=10, pady=10)
        self.end_date_entry.set_date(datetime(2000, 3, 1))

        # Rebalancing Frequency
        self.frequency_label = tk.Label(self, text="Rebalancing Frequency")
        self.frequency_label.grid(row=3, column=0, padx=10, pady=10, sticky=tk.W)
        self.frequency_combobox = ttk.Combobox(self, values=["monthly", "quarterly", "semiannually", "annually"])
        self.frequency_combobox.grid(row=3, column=1, padx=10, pady=10)
        self.frequency_combobox.set("monthly")

        # Risk-Free Rate Ticker
        self.risk_free_label = tk.Label(self, text="Risk-Free Rate Ticker (e.g., US0003M Index)")
        self.risk_free_label.grid(row=5, column=0, padx=10, pady=10, sticky=tk.W)
        self.risk_free_entry = tk.Entry(self)
        self.risk_free_entry.grid(row=5, column=1, padx=10, pady=10)
        self.risk_free_entry.insert(0, "US0003M Index")

        # Weights Type
        self.weights_label = tk.Label(self, text="Weights Type")
        self.weights_label.grid(row=6, column=0, padx=10, pady=10, sticky=tk.W)
        self.weights_combobox = ttk.Combobox(self, values=["Equally Weighted", "Max Diversification"])
        self.weights_combobox.grid(row=6, column=1, padx=10, pady=10)
        self.weights_combobox.set("Equally Weighted")

        # Run Button
        self.run_button = tk.Button(self, text="Run Backtest", command=self.run_backtest)
        self.run_button.grid(row=7, column=0, columnspan=2, pady=20)

    def run_backtest(self):
        ticker = self.ticker_entry.get()
        start_date = self.start_date_entry.get_date()
        end_date = self.end_date_entry.get_date()
        frequency = self.frequency_combobox.get()
        risk_free_rate_ticker = self.risk_free_entry.get()
        weights_type = self.weights_combobox.get()

        # Parameters for backtest
        params = {
            "currency": constant.CURRENCY,
            "start_date": start_date,
            "end_date": end_date,
            "ticker": ticker,
            "strategy": VolatilityTimingStrategy(start_date, frequency, constant.REBALANCING_MOMENT, weights_type),
            "use_pickle_universe": constant.USE_PICKLE_UNIVERSE,
            "rebalancing_frequency": frequency,
            "rebalancing_moment": constant.REBALANCING_MOMENT,
            "risk_free_rate_ticker": risk_free_rate_ticker,
            "weights_type": weights_type,
        }

        try:
            # Run the backtest
            asset_index = BackTesting.start(params)

            # Save results
            Utilities.save_data_to_pickle(asset_index, file_name="VolatilityTiming", folder_subpath="asset_indices\\monthly_eq_weighted")

            # Generate performance graphs
            asset_indices_monthly = Utilities.load_asset_indices(["LowVolatilityDecile", "MidVolatilityDecile", "HighVolatilityDecile"], 
                                                                 folder_subpath="asset_indices\\monthly_eq_weighted")
            other_data = Utilities.get_data_from_pickle('other_US_data')
           
            # Plot track records of strategies
            asset_indices_monthly = Utilities.load_asset_indices(["LowVolatilityDecile", "MidVolatilityDecile", "HighVolatilityDecile", 
                                                                  "VolatilityTiming", "VolatilityTiming2sided"],
                                                                 folder_subpath="asset_indices\\monthly_eq_weighted")
            IndexPlotter.plot_track_records(asset_indices_monthly, other_data['USRINDEX Index'])

            results = "Backtest completed successfully!"
            messagebox.showinfo("Backtest Results", results)

        except Exception as e:
            messagebox.showerror("Error", str(e))
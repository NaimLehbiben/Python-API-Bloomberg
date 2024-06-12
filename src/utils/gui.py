import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkcalendar import DateEntry
from datetime import datetime
import sys
import os
import warnings
import pandas as pd

# Add the parent directory to include 'src'
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))

# Imports after adjusting the sys.path
from src.backtester.back_tester import BackTesting
from src.strategies.strategies import (
    VolatilityTimingStrategy,
    VolatilityTimingStrategy2sided,
    LowVolatilityDecileStrategy,
    MidVolatilityDecileStrategy,
    HighVolatilityDecileStrategy
)
from src.utils.utilities import Utilities
from src.performance.graph import IndexPlotter
from src.performance.metrics import MetricsCalculator

# Import constants
import src.utils.constant as constant

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

class FinanceApp(tk.Tk):
    """Classe représentant l'application de backtesting."""
    def __init__(self):
        """Initialise l'interface graphique de l'application."""
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
        self.start_date_entry.set_date(datetime(2000, 1, 31))

        # End Date
        self.end_date_label = tk.Label(self, text="End Date")
        self.end_date_label.grid(row=2, column=0, padx=10, pady=10, sticky=tk.W)
        self.end_date_entry = DateEntry(self, date_pattern='yyyy-mm-dd')
        self.end_date_entry.grid(row=2, column=1, padx=10, pady=10)
        self.end_date_entry.set_date(datetime(2024, 2, 29))

        # Rebalancing Frequency
        self.frequency_label = tk.Label(self, text="Rebalancing Frequency")
        self.frequency_label.grid(row=3, column=0, padx=10, pady=10, sticky=tk.W)
        self.frequency_combobox = ttk.Combobox(self, values=["monthly", "quarterly", "semiannually", "annually"])
        self.frequency_combobox.grid(row=3, column=1, padx=10, pady=10)
        self.frequency_combobox.set("monthly")

        # Risk-Free Rate Ticker
        self.risk_free_label = tk.Label(self, text="Risk-Free Rate Ticker (e.g., US0003M Index)")
        self.risk_free_label.grid(row=4, column=0, padx=10, pady=10, sticky=tk.W)
        self.risk_free_entry = tk.Entry(self)
        self.risk_free_entry.grid(row=4, column=1, padx=10, pady=10)
        self.risk_free_entry.insert(0, "US0003M Index")

        # Weights Type
        self.weights_label = tk.Label(self, text="Weights Type")
        self.weights_label.grid(row=5, column=0, padx=10, pady=10, sticky=tk.W)
        self.weights_combobox = ttk.Combobox(self, values=["Equally Weighted", "Max Diversification", "Vol Scaling"])
        self.weights_combobox.grid(row=5, column=1, padx=10, pady=10)
        self.weights_combobox.set("Equally Weighted")

        # Strategy Selection
        self.strategy_label = tk.Label(self, text="Strategy")
        self.strategy_label.grid(row=6, column=0, padx=10, pady=10, sticky=tk.W)
        self.strategy_combobox = ttk.Combobox(self, values=[
            "LowVolatilityDecile",
            "MidVolatilityDecile",
            "HighVolatilityDecile",
            "VolatilityTiming",
            "VolatilityTiming2sided"
        ])
        self.strategy_combobox.grid(row=6, column=1, padx=10, pady=10)
        self.strategy_combobox.set("VolatilityTiming")

        # Bloomberg Access
        self.bloomberg_label = tk.Label(self, text="Do you have Bloomberg Access?")
        self.bloomberg_label.grid(row=7, column=0, padx=10, pady=10, sticky=tk.W)
        self.bloomberg_var = tk.BooleanVar(value=True)
        self.bloomberg_checkbutton = tk.Checkbutton(self, text="Yes", variable=self.bloomberg_var)
        self.bloomberg_checkbutton.grid(row=7, column=1, padx=10, pady=10)

        # Run Button
        self.run_button = tk.Button(self, text="Run Backtest", command=self.run_backtest)
        self.run_button.grid(row=8, column=0, columnspan=2, pady=20)

        # Treeview for displaying results
        self.results_tree = ttk.Treeview(self, columns=("Metric", "Value"), show="headings")
        self.results_tree.heading("Metric", text="Metric")
        self.results_tree.heading("Value", text="Value")
        self.results_tree.grid(row=9, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        self.grid_rowconfigure(9, weight=1)
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

    def run_backtest(self):
        """Lance le backtest avec les paramètres spécifiés par l'utilisateur."""
        ticker = self.ticker_entry.get()
        start_date = self.start_date_entry.get_date()
        end_date = self.end_date_entry.get_date()
        frequency = self.frequency_combobox.get()
        risk_free_rate_ticker = self.risk_free_entry.get()
        weights_type = self.weights_combobox.get()
        strategy_name = self.strategy_combobox.get()
        has_bloomberg = self.bloomberg_var.get()

        # Définir USE_PICKLE_UNIVERSE en fonction de l'accès à Bloomberg
        use_pickle_universe = not has_bloomberg

        # Choisir la stratégie en fonction de la sélection de l'utilisateur
        if strategy_name == "VolatilityTiming":
            strategy = VolatilityTimingStrategy(start_date, frequency, constant.REBALANCING_MOMENT, weights_type)
        elif strategy_name == "VolatilityTiming2sided":
            strategy = VolatilityTimingStrategy2sided(start_date, frequency, constant.REBALANCING_MOMENT, weights_type)
        elif strategy_name == "LowVolatilityDecile":
            strategy = LowVolatilityDecileStrategy(frequency, constant.REBALANCING_MOMENT, weights_type)
        elif strategy_name == "MidVolatilityDecile":
            strategy = MidVolatilityDecileStrategy(frequency, constant.REBALANCING_MOMENT, weights_type)
        elif strategy_name == "HighVolatilityDecile":
            strategy = HighVolatilityDecileStrategy(frequency, constant.REBALANCING_MOMENT, weights_type)
        else:
            messagebox.showerror("Error", "Invalid Strategy Selected")
            return

        # Parameters for backtest
        params = {
            "currency": constant.CURRENCY,
            "start_date": start_date,
            "end_date": end_date,
            "ticker": ticker,
            "strategy": strategy,
            "use_pickle_universe": use_pickle_universe,
            "rebalancing_frequency": frequency,
            "rebalancing_moment": constant.REBALANCING_MOMENT,
            "risk_free_rate_ticker": risk_free_rate_ticker,
            "weights_type": weights_type,
        }

        try:
            # Run the backtest
            asset_index = BackTesting.start(params)

            # Generate performance graphs
            other_data = Utilities.get_data_from_pickle('other_US_data', folder_subpath="universe")
            IndexPlotter.plot_track_records({strategy_name: asset_index}, other_data['USRINDEX Index'])

            # Calculate metrics
            metrics_calculator = MetricsCalculator(other_data, risk_free_rate_ticker)
            metrics = metrics_calculator.calculate_core_metrics(asset_index, asset_index)

            # Ensure metrics are simple numerical values and append '%'
            metrics = {k: f"{v:.2f}%" if not isinstance(v, pd.Series) else f"{v.iloc[0]:.2f}%" for k, v in metrics.items()}

            # Clear previous results
            for item in self.results_tree.get_children():
                self.results_tree.delete(item)

            # Insert new results into the Treeview
            for metric, value in metrics.items():
                self.results_tree.insert("", "end", values=(metric, value))

            asset_index.get_port_file(strategy_name)

            Utilities.save_data_to_pickle(asset_index, file_name=strategy_name, folder_subpath="asset_indices\\monthly_vol_scaling")

            results = "Backtest completed successfully!"
            messagebox.showinfo("Backtest Results", results)

        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.destroy()  # Fermer la fenêtre après avoir lancé le code
# Volatility Timing under Low-Volatility Strategy

## Objectives

This project is part of the Bloomberg API course in the Master's program in Economics & Financial Engineering at Paris Dauphine University - PSL. The aim is to replicate and analyze the methods and findings of the research article "Volatility Timing under Low-Volatility Strategy" by Poh Ling Neo and Chyng Wen Tee. This study introduces an innovative method of volatility timing under low-volatility strategy. The approach is particularly relevant for financial markets and offers new perspectives in quantitative management.

## Setting up the project

Run the file `install_for_windows.bat`, it will install dependencies and create a virtual environment for the project.

## Project Structure

Here's an overview of the project's structure:

### Description

- **data/**: Contains datasets and intermediate data files.
- **src/**: Contains all the source code for the project.
  - `backtester/`: Code related to the backtesting of strategies.
  - `base/`: Core modules and utilities.
  - `data/`: Data loading and processing scripts.
  - `performance/`: Scripts for performance metrics and graphing.
  - `strategies/`: Implementation of various strategies.
  - `utils/`: Additional utility scripts and GUI components.
- **static/**: Contains static files such as the research paper and images.
- **install_for_windows.bat**: Batch file to set up the project on Windows.
- **README.md**: The readme file.
- **requirements.txt**: Contains the list of dependencies for the project.

## Interface

The project includes a user-friendly interface designed using Tkinter. This interface allows users to interact with the different functionalities of the project, such as backtesting the strategies and computing metrics. Below is a description of the interface:

### Finance Backtesting Tool

- **Ticker**: A text input field where users can enter the Bloomberg ticker symbol of the financial instrument they wish to analyze.
- **Start Date**: A date picker allowing users to select the start date for the analysis.
- **End Date**: A date picker allowing users to select the end date for the analysis.
- **Rebalancing Frequency**: A dropdown menu where users can choose the frequency of rebalancing (e.g., monthly).
- **Risk-Free Rate Ticker**: A text input field for users to enter the Bloomberg ticker symbol of the risk-free rate.
- **Weights Type**: A dropdown menu where users can select the type of weighting strategy (Equally Weighted, Max Diversification, Vol Scaling).
- **Strategy**: A dropdown menu allowing users to select the investment strategy (e.g., Volatility Timing).
- **Do you have Bloomberg Access?**: A checkbox for users to indicate whether they have access to Bloomberg data.


This intuitive interface simplifies the process of setting up and running financial backtests, making it accessible for everyone.

## License

[MIT](https://choosealicense.com/licenses/mit/)

## Authors

Authors: **Naïm Lehbiben - Badr-Eddine El Hamzaoui**

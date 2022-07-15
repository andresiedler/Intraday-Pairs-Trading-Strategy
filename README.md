# Intraday-Pairs-Traading-Strategy

## Overview

This repository contains an implementation of an intraday pairs trading strategy applied to the brazilian market, from downloading the required data, selecting pairs of stocks through several statistical criteria, running a series of backtests simulations on a test period, to finally evaluating the results using a few different measures and techniques.

Please refer to the intraday_pairs_trading notebook for an overview and outcomes of the whole process.

## Code Organization

The whole project is coded using Python language.  

#### Description of the modules of the repository:

- import_data.py: Retrieves and preprocess the required data from the yahoo finance library and the MetaTrader5 plataform. Contains the following class:
    - ImportData
- select_pairs.py: Applies a series of statistical procedures on the train set for selecting pairs of stocks with the porpuse of implementing a pairs trading strategy. Contains the following class:
    - CointagrationApproach
- apply_strategy.py: Runs a series of backtest simulations on the test set. Contains the following classes:
    - PrepareData
    - Pair
    - AddData
    - PortfolioValue
    - TradeResult
    - ApplyStrategy
    - RunStrategy
- performance_evaluation.py: Evaluates the strategy performance uing a few different measures and techniques. Contains the following classes:
    - CSCV
    - PBO
    - SharpeAnalysis
    - StrategyRisk
    - VisualizeEquity
        
#### Description of the notebook of the repository:

- intraday_pairs_trading.ipynb: Provides a complete overview of the project, implementation and analysis of the results.
        
## Notes

- The project's complete analysis of results is presented in the repository's notebook. For re-running all steps with different inputs, the python modules and notebook should be placed in the same directory.
- The input data required for the project is collected from MetaTrader5. Many brokers offer access to this plataform for free, so it is just necessary to have an account in any of these brokers and download the plataform, which can be easily done online.

## Requirements

- backtrader==1.9.76.123
- jupyterthemes==0.20.0
- matplotlib==3.5.2
- MetaTrader5==5.0.37
- numpy==1.22.1
- pandas==1.4.2
- pytz==2021.1
- scipy==1.7.3
- seaborn==0.11.1
- statsmodels==0.13.1
- yahoo_fin==0.8.9.1
- yahooquery==2.2.15
- yfinance==0.1.70

## Install

<mark style="background-color: lightgrey">pip install -r requirements.txt</mark>

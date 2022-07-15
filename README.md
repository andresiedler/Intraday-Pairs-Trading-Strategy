# Intraday-Pairs-Traading-Strategy

## Overview

This notebook contains an implementation of an intraday pairs trading strategy applied to the brazilian market, from downloading the required data, selecting pairs of stocks through several statistical criteria, running a series of backtests simulations on a test period, to finally evaluating the results using a few different measures and techniques.

Please refer to the intraday_pairs_trading notebook for an overview and outcomes of the whole process.

## Code Organization

The whole project is coded using Python language.  

**- Description of the modules of the repository:**

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
        
**- Description of the notebook of the repository:**  

- intraday_pairs_trading.ipynb: Provides a complete overview of the project, implementation and analysis of the results.
        
## Note:

The analysis and results of this project are presented in the repository's notebook. For re-running the project with different inputs, the python modules and notebook should be placed in the same directory.

## Requirements


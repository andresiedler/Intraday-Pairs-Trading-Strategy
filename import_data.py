import numpy as np
import pandas as pd
import yahoo_fin.stock_info as ys
from datetime import datetime
import pytz
import MetaTrader5 as mt5


class ImportData:
    """
    Retrieves and preprocess data information from the brazilian index Ibovespa using the yahoo finance library and the
    software MetaTrader5.
    """

    def __init__(self,
                 interval,
                 start_train,
                 end_train,
                 start_test,
                 end_test,
                 account=int(1091371791),
                 password='belony1',
                 server='ClearInvestimentos-DEMO'):
        """
        Initializes the class and sets its parameters.

        :param interval: (str): Time interval between data points
        :param start_train: (str): Starting date of the training period
        :param end_train: (str): Final date of the training period
        :param start_test: (str): Starting date of the test period
        :param end_test: (str): Final date of the test period
        :param account: (int): MetaTrader5 account number
        :param password: (str): MetaTrader5 password
        :param server: (str): Broker information
        """

        self.interval = interval
        self.start_train = start_train
        self.end_train = end_train
        self.start_test = start_test
        self.end_test = end_test
        self.account = account
        self.password = password
        self.server = server
        self.connect()
        self.data_train = pd.DataFrame()
        self.data_test = pd.DataFrame()
        self.get_prices()
        self.sectors = None

    @staticmethod
    def get_tickers():
        """
        Downloads from yahoo finance the tickers of the stocks that compose the Ibovespa index.

        :return: (list): Tickers of the current composition of the Ibovespa index
        """

        tickers = ys.tickers_ibovespa()
        return tickers

    def connect(self):
        """Establishes connection to MetaTrader5 platform"""

        if not mt5.initialize(login=self.account, password=self.password, server=self.server):
            print("initialize() failed, error code =", mt5.last_error())
            mt5.shutdown()

        # Login
        authorized = mt5.login(login=self.account, password=self.password, server=self.server)
        if authorized:
            print("Connected to MetaTrader5")
        else:
            print("Failed to connect at account #{}, error code: {}".format(self.account, mt5.last_error()))

    def get_prices(self, threshold=50):
        """
        Downloads price information of a given set of assets.

        :param threshold: (int): Maximum number of null values of each stock for removing it from the sample set
        :return: (pd.DataFrame): DataFrame containing stock prices in each column
        """

        self.data_test = pd.DataFrame()
        self.data_train = pd.DataFrame()
        tickers = self.get_tickers()
        for i in tickers:
            rates = mt5.copy_rates_range(i, self.interval, self.start_train, self.end_train)
            if rates is not None:
                df = pd.DataFrame(rates)
                self.data_train[i] = df['close']
            else:
                print('Asset not found: ', i)

        for i in tickers:
            rates = mt5.copy_rates_range(i, self.interval, self.start_test, self.end_test)
            if rates is not None:
                df = pd.DataFrame(rates)
                self.data_test[i] = df['close']
            else:
                continue

        # Remove stocks from the DataFrame that contain more null values than a given threshold
        null_values = self.data_train.isnull().sum()
        null_values = null_values[null_values > 0]
        remove_nulls = list(null_values[null_values > threshold].index)
        self.data_train = self.data_train.drop(columns=remove_nulls)
        self.data_train.dropna(inplace=True, axis=0)

        null_values = self.data_test.isnull().sum()
        null_values = null_values[null_values > 0]
        remove_nulls = list(null_values[null_values > threshold].index)
        self.data_test = self.data_test.drop(columns=remove_nulls)
        self.data_test.dropna(inplace=True, axis=0)
        indice = pd.DataFrame(mt5.copy_rates_range(tickers[0], self.interval, self.start_test, self.end_test))
        self.data_test['time'] = indice['time']
        self.data_test['time'] = pd.to_datetime(self.data_test['time'], unit='s')
        # self.data_test.set_index('time', inplace=True)

        return self.data_train, self.data_test

    @staticmethod
    def log_prices(price_data):
        """
        Calculates the log-prices of a given price data set.

        :param price_data: (pd.DataFrame): DataFrame containing stock prices in each column
        :return: (pd.DataFrame): DataFrame containing stock log-prices in each column
        """

        log_prices = np.log(price_data)
        return log_prices

    @staticmethod
    def returns(price_data):
        """
        Calculates the returns of a given stock price data set.

        :param price_data: (pd.DataFrame): DataFrame containing stock prices in each column
        :return: (pd.DataFrame): DataFrame containing stock returns in each column
        """

        returns = price_data.pct_change()
        returns = returns.iloc[1:]
        return returns

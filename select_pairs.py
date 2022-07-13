from dataclasses import dataclass
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
import statsmodels.tsa.vector_ar.vecm as vm
import statsmodels.api as sm

warnings.filterwarnings('ignore')


@dataclass
class CointegrationApproach:
    """
    Implements a series of statistical procedures to select pairs of assets from a universe of stocks with the purpose
    of applying a pairs trading strategy.

    - Steps followed:
        1. Stationarity test of each asset from the universe considered
        2. Select assets that show evidence of being integrated of order 1
        3. Group the selected assets by sector
        4. Pairwise cointegration tests (Engle-Granger and Johansen) of all the permutations in the assets' sectors
        5. Select the pairs that reach the desired statistical significance on the cointegration test
        6. Calculate the hedge ratios of the selected pairs through linear regression
        7. Calculate the spreads of the selected pairs using the hedge ratios as the coefficient parameter
        8. Calculate the Hurst Exponent of each spread
        9. Calculate the half-life of each spread
        10. Calculate the number of zero-crossings of each spread
        11. Select the pairs that meet the previously defined criteria
        12. Calculate the statistics of interest to apply a pairs trading strategy of each pair on the test period

    Sources:
        - Sarmento, Simão Moraes, and Nuno Horta. "Enhancing a pairs trading strategy with the application of machine
          learning." Expert Systems with Applications 158 (2020): 113490.
        - Krauss, Christopher. "Statistical arbitrage pairs trading strategies: Review and outlook." Journal of Economic
          Surveys 31.2 (2017): 513-545.
        - Chan, Ernest P. Quantitative trading: how to build your own algorithmic trading business. John Wiley & Sons,
          2021.
        - Chan, Ernest P. Algorithmic trading: winning strategies and their rationale. Vol. 625. John Wiley & Sons,
          2013.
    """

    data_train: pd.DataFrame
    data_test: pd.DataFrame
    p_value_threshold: float = 0.10
    hurst_threshold: float = 0.5
    min_half_life: int = 5
    max_half_life: int = 50
    min_zero_crossings: int = 3

    @staticmethod
    def stationarity_test(price_series):
        """
        Performs Augmented Dickey-Fuller unit root test to check evidence of stationarity of a given price series.

        :param price_series: (pd.Series): Stock price series.
        :return: (float): P-value of the Augmented Dickey-Fuller test statistic.
        """
        pvalue_adf = adfuller(price_series)[1]
        return pvalue_adf

    def select_integrated1(self):
        """
        Method to select stocks that are integrated of order 1 to proceed with cointegration test on the following
        steps.

        :return: (pd.DataFrame): Pandas DataFrame with stocks prices series that provided evidence of being
        integrated of order 1 in each column.
        """
        integrated1_assets = []
        # integrated1_assets = pd.DataFrame(self.data_train)
        for i in self.data_train:
            pvalue = self.stationarity_test(self.data_train['{}'.format(i)])
            if pvalue >= self.p_value_threshold:
                integrated1_assets.append(i)
        return integrated1_assets

    def get_sectors(self):
        """
        Provides the sector of each stock acquired from yahoo finance.

        :return: (list): List of stocks and their respective economy sector.
        """
        sectors_list = []
        stocks_list = self.select_integrated1()
        for i in stocks_list:
            info = yf.Ticker('{}.SA'.format(i)).info
            sector = info.get('sector')
            sectors_list.append([i, sector])
        return sectors_list

    @staticmethod
    def calculate_hurst_exponent(time_series):
        """
        Calculates the Hurst Exponent of a given time series.
            - The sublinear function of time (as if the series is stationary) can be approximated by: tau^(2H), where
              tau is the time separating two measurements and H is the Hurst Exponent.
            - Hypothesis Test:

                - Hurst Exponent < 0.5: Time series is stationary.
                - Hurst Exponent = 0.5: Time series is a geometric random walk.
                - Hurst Exponent > 0.5: Time series is trending.

        :param time_series: (array or Series): Time series of the spread of two assets' prices.
        :return: (float): Hurst Exponent of a given time series.
        """
        lags = range(2, 100)
        tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log(lags), np.log(tau), 1)

        return poly[0] * 2.0

    @staticmethod
    def calculate_half_life(time_series):
        """
        Calculates the half-life of a given time series.
            - Measures how quickly a time series reverts to its mean.
            - Good predictor of the profitability or Sharpe ratio of a mean-reverting strategy.

        :param time_series: (array or Series): Time series of the spread of two assets' prices.
        :return: Half-life of a given time series.
        """
        time_series_lag = np.roll(time_series, 1)
        time_series_lag[0] = 0
        ret = time_series - time_series_lag
        ret[0] = 0
        time_series_lag2 = sm.add_constant(time_series_lag)
        model = sm.OLS(ret[1:], time_series_lag2[1:])
        res = model.fit()

        return -np.log(2) / res.params[1]

    @staticmethod
    def calculate_zero_crossings(time_series):
        """
        Calculates the number of times a given time series crosses zero.

        :param time_series: (array or Series): Time series of the spread of two assets' prices.
        :return: Number of times the time series crosses zero.
        """
        x = time_series - time_series.mean()
        zero_cross = sum(1 for i, _ in enumerate(x) if (i + 1 < len(x)) if ((x[i] * x[i + 1] < 0) or (x[i] == 0)))

        return zero_cross

    def statistical_analysis(self, pair):
        """
        Receives a candidate pair and performs a series of statistical procedures to check if the given pair has the
        desired properties for applying particular trading strategy.
        Statistical analysis and calculations applied to the pair of stocks:
            - Cointegration tests (Engle-Granger and Johansen)
            - Hedge-ratio
            - Spread
            - Hurst Exponent
            - Half-life
            - Number of zero crossings

        :param pair: (Tuple[str, str]): Eligible pair of stocks.
        :return: (list): Statistical analysis and calculations results.
        """
        S1_train = np.asarray(pair[0])
        S2_train = np.asarray(pair[1])
        S1_c = sm.add_constant(S1_train)

        # Cointegration test (perform Engle-Granger test artificially - coint from statsmodels not working as expected)
        linear_regression = sm.OLS(S2_train, S1_c)
        linear_regression_result = linear_regression.fit()
        hr = linear_regression_result.params[0]
        spread = pair[1] - hr * pair[0]
        coint_engle_granger = self.stationarity_test(spread)

        # Cointegration test (Johansen)
        merge_assets = pd.concat([pair[1], pair[0]], axis=1)
        coint_johansen_test = vm.coint_johansen(merge_assets.values, det_order=0, k_ar_diff=1)
        significance_position = {90: 0, 95: 1, 99: 2}
        significance = significance_position[int(100 - (self.p_value_threshold * 100))]

        # Recover the hedge-ratio
        hedge_ratio = linear_regression_result.params[0]

        # Calculate the pair's spread
        spread = pair[1] - hedge_ratio * pair[0]
        spread_array = np.asarray(spread)

        # Calculate the spread's Hurst Exponent
        hurst_exponent = self.calculate_hurst_exponent(spread_array)

        # Calculate the spread's half-life
        half_life = self.calculate_half_life(spread_array)

        # Calculate the spread's number of zero crossings
        zero_crossings = self.calculate_zero_crossings(spread_array)

        if coint_engle_granger < self.p_value_threshold and \
                coint_johansen_test.lr1[0] > coint_johansen_test.cvt[0, significance] and \
                hurst_exponent < self.hurst_threshold and \
                (half_life >= self.min_half_life) and \
                (half_life < self.max_half_life) and \
                zero_crossings >= self.min_zero_crossings:
            tests_result = 'Criteria met'

        else:
            tests_result = 'Criteria not met'

        return [tests_result,
                coint_engle_granger,
                coint_johansen_test.lr1[0],
                coint_johansen_test.cvt[0, significance],
                hurst_exponent,
                half_life,
                zero_crossings,
                hedge_ratio,
                spread]

    def test_set_statistics(self, results):
        """
        Receives a previously analyzed and selected pair and calculates the required statistics for applying in the
        following steps at the strategy application.

        :param results: (list): Statistical analysis and calculations results of the previously selected pairs.
        :return: (pd.DataFrame): DataFrame with the required data information of the pair for later usage on
        strategy application.
        """

        # Calculate the pair's spread on the test set
        results['spread_test'] = results['Y_test'] - results['hedge_ratio'] * results['X_test']

        # Calculate the pair's z-score on the test set
        results['zscore'] = (results['spread_test'] - results['spread_test'].mean()) / np.std(results['spread_test'])

        # Create DataFrame with the required statistics for later usage on strategy application
        asset1 = np.zeros(results['zscore'].shape)
        asset1 = np.where(asset1 == 0., results['asset1'], asset1)
        asset2 = np.zeros(results['zscore'].shape)
        asset2 = np.where(asset2 == 0., results['asset2'], asset2)
        hedge_ratio = np.zeros(results['zscore'].shape)
        hedge_ratio = np.where(hedge_ratio == 0., results['hedge_ratio'], hedge_ratio)
        period = range(1, int(results['zscore'].shape[0]) + 1)
        pair_data = pd.DataFrame(index=self.data_test['time'])
        pair_data['asset1'] = asset1
        pair_data['asset2'] = asset2
        pair_data['hedge_ratio'] = hedge_ratio
        pair_data['zscore'] = results['zscore'].values
        pair_data['period'] = period

        return pair_data

    def select_pairs(self):
        """
        Receives the asset price data and retrieves the economy sector of each stock
        select the pairs with the desired properties for implement the trading strategy.
        Within each cluster, select pairs using a few statistical criteria, namely:
            - Cointegration (Engle-Granger and Johansen)
            - Hurst Exponent
            - Half-life
            - Zero crossings
        Returns the pairs that meet the desired criteria for implement the trading strategy, the assets that incorporate
        the pairs and the statistical information of each pair.

        :return: (list): Pairs selected under the chosen criteria.
        :return: (list): Assets that embody at least one selected pair.
        :return: (pd.DataFrame): Statistical information of each pair.
        """

        # Get the sector of every selected asset (integrated of order 1)
        list_sectors = self.get_sectors()

        selected_pairs = []
        stats_test = []
        df_stats_test = pd.DataFrame()
        all_sectors = []
        stocks = []
        for i in list_sectors:
            stocks.append(i[0])
            all_sectors.append(i[1])
        sectors = set(all_sectors)
        n_sectors = len(np.unique(all_sectors))
        sectors_series = pd.Series(all_sectors, index=stocks)

        # Iterate over each sector
        for sector in sectors:
            symbols = sectors_series[sectors_series == sector].index
            train_series = self.data_train[symbols]
            test_series = self.data_test[symbols]
            n = train_series.shape[1]
            keys = train_series.keys()
            print('Searching for pairs in sector: ', sector)

            # Iterate over every permutation of assets to form a pair within each sector
            for i in range(n):
                for j in range(i + 1, n):
                    pairs_train = [(train_series[keys[i]], train_series[keys[j]]),
                                   (train_series[keys[j]], train_series[keys[i]])]
                    pairs_keys = [(keys[i], keys[j]), (keys[j], keys[i])]
                    pairs_test = [(test_series[keys[i]], test_series[keys[j]]),
                                   (test_series[keys[j]], test_series[keys[i]])]

                    statistical_results1 = self.statistical_analysis(pairs_train[0])
                    statistical_results2 = self.statistical_analysis(pairs_train[1])

                    if statistical_results1[0] == 'Criteria met' and statistical_results2[0] == 'Criteria met':
                        print('Pair found: Both combinations meet the criteria:', [pairs_keys[0], pairs_keys[1]])
                        if statistical_results1[1] < statistical_results2[1]:
                            result = {'asset1': pairs_keys[0][0],
                                      'asset2': pairs_keys[0][1],
                                      'p_value': statistical_results1[1],
                                      'hedge_ratio': statistical_results1[7],
                                      'hurst_exponent': statistical_results1[4],
                                      'half_life': int(statistical_results1[5]),
                                      'zero_cross': statistical_results1[6],
                                      'spread': statistical_results1[8],
                                      'X_train': pairs_train[0][0],
                                      'Y_train': pairs_train[0][1],
                                      'X_test': pairs_test[0][0],
                                      'Y_test': pairs_test[0][1],
                                      }
                            selected_pairs.append((result['asset1'], result['asset2']))
                            pair_df = self.test_set_statistics(result)
                            stats_test.append(pair_df)

                        else:
                            result = {'asset1': pairs_keys[1][0],
                                      'asset2': pairs_keys[1][1],
                                      'p_value': statistical_results2[1],
                                      'hedge_ratio': statistical_results2[7],
                                      'hurst_exponent': statistical_results2[4],
                                      'half_life': int(statistical_results2[5]),
                                      'zero_cross': statistical_results2[6],
                                      'spread': statistical_results2[8],
                                      'X_train': pairs_train[1][0],
                                      'Y_train': pairs_train[1][1],
                                      'X_test': pairs_test[1][0],
                                      'Y_test': pairs_test[1][1],
                                      }
                            selected_pairs.append((result['asset1'], result['asset2']))
                            pair_df = self.test_set_statistics(result)
                            stats_test.append(pair_df)

                    elif statistical_results1[0] == 'Criteria met' and statistical_results2[0] == 'Criteria not met':
                        print('Pair found:', pairs_keys[0])
                        result = {'asset1': pairs_keys[0][0],
                                  'asset2': pairs_keys[0][1],
                                  'p_value': statistical_results1[1],
                                  'hedge_ratio': statistical_results1[7],
                                  'hurst_exponent': statistical_results1[4],
                                  'half_life': int(statistical_results1[5]),
                                  'zero_cross': statistical_results1[6],
                                  'spread': statistical_results1[8],
                                  'X_train': pairs_train[0][0],
                                  'Y_train': pairs_train[0][1],
                                  'X_test': pairs_test[0][0],
                                  'Y_test': pairs_test[0][1],
                                  }
                        selected_pairs.append((result['asset1'], result['asset2']))
                        pair_df = self.test_set_statistics(result)
                        stats_test.append(pair_df)

                    elif statistical_results1[0] == 'Criteria not met' and statistical_results2[0] == 'Criteria met':
                        print('Pair found:', pairs_keys[1])
                        result = {'asset1': pairs_keys[1][0],
                                  'asset2': pairs_keys[1][1],
                                  'p_value': statistical_results2[1],
                                  'hedge_ratio': statistical_results2[7],
                                  'hurst_exponent': statistical_results2[4],
                                  'half_life': int(statistical_results2[5]),
                                  'zero_cross': statistical_results2[6],
                                  'spread': statistical_results2[8],
                                  'X_train': pairs_train[1][0],
                                  'Y_train': pairs_train[1][1],
                                  'X_test': pairs_test[1][0],
                                  'Y_test': pairs_test[1][1],
                                  }
                        selected_pairs.append((result['asset1'], result['asset2']))
                        pair_df = self.test_set_statistics(result)
                        stats_test.append(pair_df)

                    elif statistical_results1[0] == 'Criteria not met' and statistical_results2[0] == 'Criteria not met':
                        continue

            if not stats_test:
                continue

            else:
                df_stats_test = pd.concat(stats_test)
                df_stats_test = df_stats_test.set_index(pd.DatetimeIndex(df_stats_test.index))
                df_stats_test.sort_values('period', inplace=True)

        print('Total number of sectors analyzed:', n_sectors)
        print('Found {} pairs'.format(len(selected_pairs)))
        unique_assets = np.unique([(element[0], element[1]) for element in selected_pairs])
        print('The pairs contain {} unique assets'.format(len(unique_assets)))
        if len(selected_pairs) <= 1:
            raise Exception("Number of selected pairs must be greater than 1 for execution of the strategy")

        return selected_pairs, unique_assets, df_stats_test

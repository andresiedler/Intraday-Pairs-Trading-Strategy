from dataclasses import dataclass
from pathlib import Path
import csv
import numpy as np
import pandas as pd
import backtrader as bt
import warnings
import datetime

warnings.filterwarnings('ignore')


@dataclass
class PrepareData:
    """
    Process stocks and pairs related data of the test period for later usage on strategy implementation.
    """

    trades_data: pd.DataFrame
    prices_data: pd.DataFrame

    def prepare_data(self):
        """
        Receives stocks prices series and pairs' data information of the test period and process it to feed the
        pairs trading strategy.

        :return: (np.array): Array of stocks that compose at least one selected pair
        :return: (pd.DataFrame): Pairs data information for strategy application
        :return: (pd.DataFrame): Stocks prices series information for strategy application
        """

        # self.trades_data.set_index('time', inplace=True)
        trades_data = self.trades_data.set_index(pd.DatetimeIndex(self.trades_data.index))
        trade_dates = np.unique(trades_data.index)
        start = trade_dates.min()
        end = trade_dates.max()
        traded_symbols = trades_data.asset1.append(trades_data.asset2).unique()
        trades_data.sort_values('period', inplace=True)

        index_slice = pd.IndexSlice
        self.prices_data.set_index('time', inplace=True)
        self.prices_data = self.prices_data.set_index(pd.DatetimeIndex(self.prices_data.index))
        prices_data = pd.DataFrame(self.prices_data.stack(), columns={'close'})
        indice = prices_data.index
        indice.rename(['time', 'symbol'], inplace=True)
        prices_data = prices_data.sort_index().loc[index_slice[str(start):str(end), traded_symbols], :]
        prices_data = prices_data.reorder_levels(['symbol', 'time'])

        return traded_symbols, trades_data, prices_data


@dataclass
class Pair:
    """
    Class that stores pairs' information required along the implementation of the pairs trading strategy.
    """

    period: int
    asset1: str
    asset2: str
    size1: float
    size2: float
    long: bool
    hr: float
    price1: float
    price2: float
    pos1: float
    pos2: float

    def calculate_position(self, price1, price2):
        """
        Method to calculate the market value of a pair.

        :param price1: Price of the first stock that is part of the pair
        :param price2: Price of the second stock that is part of the pair
        :return: Pair's market value
        """

        return price1 * self.size1 + price2 * self.size2


class AddData(bt.feeds.PandasData):
    """
    Class that process and feeds a pandas DataFrame to Backtrader's (backtesting library) strategy executioner method
    for accomplishment of the strategy simulation.
    """

    linesoveride = True
    cols = ['close']

    # Create lines
    lines = tuple(cols)

    # Configure timeframe
    timeframe = bt.TimeFrame.Minutes

    # Configure parameters
    params = {c: -1 for c in cols}
    params.update({'datetime': None})
    params = tuple(params.items())


class PortfolioValue(bt.analyzers.Analyzer):
    """
    Backtrader analyzer that provides the total portfolio market value at every point in time at a given time interval.
    """

    def __init__(self):
        self.portfolio_values = []

    def start(self):
        super(PortfolioValue, self).start()

    def next(self):
        portfolio_value = cerebro.broker.getvalue()
        self.portfolio_values.append(portfolio_value)

    def get_analysis(self):
        return self.portfolio_values


class TradeResult(bt.analyzers.Analyzer):
    """
    Backtrader analyzer that provides the net profit (accounting for commissions and costs) of every stock's pair closed
    position.
    """
    def __init__(self):
        self.pnls = []

    def start(self):
        super(TradeResult, self).start()

    def notify_order(self, order):
        if order.executed:
            # Calculate the trade PnL when closing a pairs' position (x2 for opening and closing the position)
            if order.executed.pnl != 0:
                self.pnls.append(order.executed.pnl - (order.executed.comm * 2))

    def get_analysis(self):
        return self.pnls


class ApplyStrategy(bt.Strategy):
    """
    Class that creates a mean-reverting strategy logic using Backtrader backtesting library.
    Strategy information:
        - Pairs trading intraday strategy analyzing market conditions at a particular fixed time interval.
        - Searches for market opportunities from a predefined start session time, set at 10:30 am (30 minutes after the
          actual market opening), until the end of session time, set at 16:00 (1 hour before the actual market
          closing).
        - From the end session time until the actual close market time all the positions are closed at market value,
          therefore no positions are held overnight (rental costs from short positions are not generated).
        - The strategy algorithm takes a long (short) position on a pair if the spread of the pair deviates from its
          normalized mean (equals zero by construction) more than the long_entry (short_entry) parameter and closes the
          pair's position if the spread of the pair crosses the long_exit (short_exit) parameter.
        - The trading algorithm only takes a position on a pair if no positions are currently opened for that pair.
        - The portfolio positions are set to be equally weighted across the pairs (pair position = 1 / total number of
          pairs)

    Sources:
        - Jansen, Stefan. Machine Learning for Algorithmic Trading: Predictive models to extract signals from market and
          alternative data for systematic trading strategies with Python. Packt Publishing Ltd, 2020.
        - Pairs Trading In Brazil And Short Straddles In The US Market. Available at
          https://blog.quantinsti.com/algo-trading-epat-projects-12-april-2022/.
    """

    params = (('trades', None),
              ('long_entry', -1.),
              ('short_entry', 1.),
              ('long_exit', -0.5),
              ('short_exit', 0.5),
              ('verbose', True),
              ('log_file', 'backtest.csv'),
              ('number_pairs', None),
              ('start_session', datetime.time(10, 30, 00)),
              ('end_session', datetime.time(16, 30, 00)),
              ('close_positions_time', datetime.time(16, 00, 00)))

    def __init__(self):
        """
        Initializes the class attributes.
        """

        self.bar_executed = None
        self.buy_comm = None
        self.buy_price = None
        self.target_value = None
        self.active_pairs = {}
        self.closing_pairs = {}
        self.last_close = {}
        self.current_date = None
        self.order_status = dict(enumerate(['Created', 'Submitted', 'Accepted', 'Partial', 'Completed',
                                            'Canceled', 'Expired', 'Margin', 'Rejected']))

    def save_log(self):
        """
        Saves the simulated backtests on a csv file for a clearer and more concise printed information.
        """
        if Path(self.p.log_file).exists():
            Path(self.p.log_file).unlink()
        with Path(self.p.log_file).open('a') as f:
            log_writer = csv.writer(f)
            log_writer.writerow(
                ['Date', 'Pair', 'Symbol', 'Order #', 'Reason',
                 'Status', 'Long', 'Price', 'Size', 'Position'])

    def log(self, txt, dt=None):
        """
        Defines the simulated backtests logging parameters.

        :param txt: (any): Backtest information.
        :param dt: (datetime.datetime): Date and time of the current backtest information.
        :return: Backtest log at the required format.
        """
        dt = dt or self.datas[0].datetime.datetime(0)
        with Path(self.p.log_file).open('a') as f:
            log_writer = csv.writer(f)
            log_writer.writerow([dt.isoformat()] + txt.split(','))

    @staticmethod
    def get_pair_id(asset1, asset2):
        """
        Provides the pair's identification. There will not be two different pairs with the positions exchanged, however,
        the order of the stocks matters for some calculations, like order size for instance.

        :param asset1: (str): First stock of the pair.
        :param asset2: (str): Second stock of the pair.
        :return: (str): Pair's identification.
        """

        return f'{asset1}.{asset2}'

    def notify_order(self, order):
        """
        Tracks the status of every submitted order to Backtarder's broker environment, since its creation until the
        execution or rejection.

        :param order: (backtrader.order.Order): Translate the decisions made on the strategy's logic to Backtarder's
        broker environment to execute a particular action.
        """

        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f, Pnl %.2f, Commission %.2f, Net %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm,
                          order.executed.pnl,
                          order.executed.comm,
                          order.executed.pnl - order.executed.comm))

            elif order.issell():
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f, Pnl %.2f, Commission %.2f, Net %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm,
                          order.executed.pnl,
                          order.executed.comm,
                          order.executed.pnl - order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

    def notify_trade(self, trade):
        """
        Keeps track of every closed trade along the strategy's simulation execution.

        :param trade: (backtrader.trade.Trade): A trade is opened when a position in a particular asset goes from zero
        to any value different from zero (positive or negative) and is closed when a position goes from any value
        different from zero back to zero.
        """

        if not trade.isclosed:
            return

        self.log('TRADE CLOSED - OPERATION PROFIT, GROSS %.2f, NET %.2f' % (trade.pnl, trade.pnlcomm))

    def enter_pair(self, asset1, asset2, hr, period, long=True):
        """
        Method to take a market position on a pair (long or short).

        :param asset1: (str): First stock that constitutes the pair.
        :param asset2: (str): Second stock that constitutes the pair.
        :param hr: (float): Hedge-ratio of the pair.
        :param period: (int): Current period
        :param long: (bool): Flag to indicate a long or a short position on a pair.
        """
        pair_id = self.get_pair_id(asset1, asset2)
        price1 = self.last_close[asset1]
        price2 = self.last_close[asset2]
        size2 = self.target_value / price2
        size1 = hr * size2

        # Pairs' long position: spread < -zscore
        if long:
            pair = Pair(asset1=asset1, asset2=asset2, period=period, size1=size1, size2=size2, pos1=price1 * size1,
                        pos2=price2 * size2, hr=hr, long=long, price1=price1, price2=price2)
            order2 = self.buy(data=asset2, size=size2)
            order1 = self.sell(data=asset1, size=abs(size1))
            self.active_pairs[pair_id] = pair
            self.log(f'{pair_id}, {asset1}, {order1.ref}, {price1}, {size1}: Long pair - Sell order created')
            self.log(f'{pair_id}, {asset2}, {order2.ref}, {price2}, {size2}: Long pair - Buy order created')

        # Pairs' short position: spread > zscore
        elif not long:
            pair = Pair(asset1=asset1, asset2=asset2, period=period, size1=size1, size2=size2, pos1=price1 * size1,
                        pos2=price2 * size2, hr=hr, long=long, price1=price1, price2=price2)
            order2 = self.sell(data=asset2, size=abs(size2))
            order1 = self.buy(data=asset1, size=abs(size1))
            self.active_pairs[pair_id] = pair
            self.log(f'{pair_id}, {asset1}, {order1.ref}, {price1}, {size1}: Short pair - Buy order created')
            self.log(f'{pair_id}, {asset2}, {order2.ref}, {price2}, {size2}: Short pair - Sell order created')

    def exit_pair(self, pair, pair_id):
        """
        Method to close a pair's market position (long or short).

        :param pair: (Pair): Class that holds a series of information about a pair.
        :param pair_id: (str): Pair identification.
        """

        self.close(data=pair.asset1, size=abs(pair.size1))
        self.close(data=pair.asset2, size=abs(pair.size2))
        self.closing_pairs[pair_id] = pair

        self.log(f'{pair_id}, {pair.asset1}, {pair.long}, {pair.size1}: Close order created')
        self.log(f'{pair_id}, {pair.asset2}, {pair.long}, {pair.size2}: Close order created')

    def prenext(self):
        """
        Method called before the minimum period of all datas/indicators have been meet for the Backtarder's strategy
        logic to start executing, so any required information can be accessed before achieving that defined minimum
        period.
        """

        self.next()

    def next(self):
        """
        Method called every data point when the minimum period for all datas/indicators have been meet.
        """

        # Check if the current date has any trades to be made
        self.current_date = pd.Timestamp(self.datas[0].datetime.datetime())
        if self.current_date not in self.p.trades.index:
            return

        # Calculate pairs' market positions target values (equally weighted)
        portfolio_value = self.broker.get_value()
        target = 1 / len(self.p.number_pairs)
        self.target_value = portfolio_value * target

        # Get trades of the current analyzed date and apply strategy logic
        self.last_close = {d._name: d.close[0] for d in self.datas}
        transactions = self.p.trades.loc[self.current_date]
        for asset1, asset2, zscore, hr, period in zip(transactions.asset1,
                                                      transactions.asset2,
                                                      transactions.zscore,
                                                      transactions.hedge_ratio,
                                                      transactions.period):

            pair_id = self.get_pair_id(asset1, asset2)

            if self.p.start_session <= self.datas[0].datetime.time() < self.p.end_session:

                if self.p.long_exit < zscore <= self.p.short_exit:
                    if pair_id not in self.active_pairs:
                        self.log(f'{pair_id}: Pair not active - Signal to take position not triggered')
                    elif pair_id in self.active_pairs:
                        self.log(f'{pair_id}: Pair active - Signal to close position triggered')
                        pair = self.active_pairs.pop(pair_id, None)
                        self.exit_pair(pair, pair_id)

                if zscore < self.p.long_entry:
                    if pair_id in self.active_pairs:
                        self.log(f'{pair_id}: Pair already active')
                    elif pair_id not in self.active_pairs:
                        self.log(f'{pair_id}: Pair not active - Taking long position on pair')
                        self.enter_pair(asset1,
                                        asset2,
                                        hr,
                                        period,
                                        long=True)

                if zscore > self.p.short_entry:
                    if pair_id in self.active_pairs:
                        self.log(f'{pair_id}: Pair already active')
                    elif pair_id not in self.active_pairs:
                        self.log(f'{pair_id}: Pair not active - Taking short position on pair')
                        self.enter_pair(asset1,
                                        asset2,
                                        hr,
                                        period,
                                        long=False)

            elif self.datas[0].datetime.time() >= self.p.close_positions_time:
                if len(self.active_pairs) > 0:
                    self.log(f'End of session, active pairs need to be closed')
                if not self.active_pairs:
                    self.log(f'End of session, no active pairs at the moment')
                if pair_id in self.active_pairs:
                    self.log(f'{pair_id}: Pair active - Closing pairs position before the end of the day')
                    pair = self.active_pairs.pop(pair_id, None)
                    self.exit_pair(pair, pair_id)


@dataclass
class RunStrategy:
    """
    Class that executes the pairs trading strategies simulations using the Backtrader backtesting library.

    Sources:
        - JUNIOR, F. A. D. L. Expoente DE Hurst e sua Eficácia em Estratégias de Pairs Trading Intradiário no Mercado
          Brasileiro. Fundação Getulio Vargas, São Paulo, 2019.
        - Pontuschka, M., & Perlin, M. (2015). A estratégia de pares no mercado acionário brasileiro: O Impacto da
          frequência de dados. Revista de Administração Mackenzie, 16(2), 188-213.
        - Singh, A., Pachanekar, R. Sharpe Ratio: Calculation, Application, Limitations - Quantinsti blog, available at
          https://blog.quantinsti.com/sharpe-ratio-applications-algorithmic-trading/
        - Pei, H., The Correct Vectorized Backtest Methodology for Pairs Trading, available at
          https://hudsonthames.org/correct-backtest-methodology-pairs-trading/
    """

    trades: pd.DataFrame
    prices: pd.DataFrame
    traded_assets: list
    n_pairs: list
    cash: int = 100000
    commission: float = 0.0004

    def run_strategy(self):
        """
        Method that loops over a range of parameters and runs every simulation according to the previously defined
        strategy logic, storing the necessary data for evaluation of the trading strategy performance.
        Strategy execution information:
            - Backtests simulations are executed varying the strategy parameters (long entry, short entry, long exit and
              short exit), which aims to create empirical distribution for each statistic of interest, instead of just a
              point estimate.
            - Transactions costs: There are no brokerage fees, so the transactions costs considered are from B3
              (brazilian stock exchange) and slippage. Following (Pontuschka & Perlin, 2015) and (Junior, 2019), the
              total cost per transaction is set at 0,04%.

        :return: (pd.DataFrame): Portfolio market value of every backtest simulation at every point in time.
        :return: (pd.DataFrame): Portfolio returns of every backtest simulation at every point in time.
        :return: (list): Annualized Sharpe ratios of every simulation.
        :return: (list): Non-annualized Sharpe ratios of every simulation.
        :return: (list): Total number of trades of every simulation.
        :return: (pd.DataFrame): Net result (profit or loss) of every trade for all the simulations.
        """

        # Define the necessary lists, DataFrames and parameters ranges
        sharpes_list = []
        ann_sharpes_list = []
        number_trades = []
        pf_list = []
        trades_results = []
        long_entries = np.arange(-1.5, -0.75, 0.25)
        short_entries = np.arange(1., 1.75, 0.25)
        long_exits = np.arange(-0.75, 0.0, 0.25)
        short_exits = np.arange(0.25, 1., 0.25)

        # Loop over entry and exit parameters
        for i in long_entries:
            for j in short_entries:
                for k in long_exits:
                    for q in short_exits:

                        # Configure Cerebro
                        global cerebro
                        cerebro = bt.Cerebro()
                        cerebro.broker.set_coc(True)
                        cerebro.broker.setcash(self.cash)
                        cerebro.broker.setcommission(self.commission)

                        # Add data
                        idx = pd.IndexSlice
                        for symbol in self.traded_assets:
                            dataframe = self.prices.loc[idx[symbol, :], :].droplevel('symbol', axis=0)
                            dataframe.index.name = 'datetime'
                            bt_data = AddData(dataname=dataframe)
                            cerebro.adddata(bt_data, name=symbol)

                        # Add strategy
                        strategy_trades = self.trades
                        number_pairs = self.n_pairs
                        cerebro.addstrategy(ApplyStrategy,
                                            number_pairs=number_pairs,
                                            trades=strategy_trades,
                                            verbose=True,
                                            long_entry=i,
                                            short_entry=j,
                                            long_exit=k,
                                            short_exit=q)

                        # Add analyzers (PortfolioValue and TradeClosed classes)
                        cerebro.addanalyzer(PortfolioValue, _name="portfolio_value")
                        cerebro.addanalyzer(TradeResult, _name="trade_result")

                        # Run strategy
                        results = cerebro.run(maxcups=1,
                                              live=False,
                                              runonce=True,
                                              exactbars=False,
                                              optdatas=True,
                                              optreturn=True,
                                              stdstats=False,
                                              quicknotify=True)

                        # Analysis of the results: Calculate annualized Sharpe ratio and total number of trades
                        trial_pnls = np.array(results[0].analyzers.getbyname("trade_result").get_analysis())
                        trades_results.append(trial_pnls)
                        strategy_duration = np.busday_count(strategy_trades.index[0].date(), strategy_trades.index[-1].date())
                        trades_per_year = (len(trial_pnls) / strategy_duration) * 252
                        ann_trial_sr = (trades_per_year * trial_pnls.mean()) / (np.sqrt(trades_per_year * trial_pnls.var()))
                        trial_sr = trial_pnls.mean() / np.sqrt(trial_pnls.var())
                        ann_sharpes_list.append(ann_trial_sr)
                        sharpes_list.append(trial_sr)
                        number_trades.append(len(trial_pnls))
                        pf_list.append(results[0].analyzers.getbyname("portfolio_value").get_analysis())

                        msg = f'Sharpe ratio: {ann_trial_sr:3,.4f} | '
                        msg += f'Number of trades: {len(trial_pnls):3,.0f} | '
                        msg += f'Parameters: Long[{i}, {k}] - Short[{j}, {q}] | '
                        msg += f'Final Portfolio Value: {cerebro.broker.getvalue():3,.2f}'
                        print(msg)

        # Create Trials Pnls and Returns DataFrame
        df_trades_results = pd.DataFrame(trades_results).T
        df = pd.DataFrame(pf_list).T
        old_columns = list(df)
        new_col_list = []
        for i in range(1, len(df.columns) + 1):
            item = 'trial{}'.format(i)
            new_col_list.append(item)
        df.rename(columns={old_columns[idx]: name for (idx, name) in enumerate(new_col_list)}, inplace=True)

        rets = (df / df.shift(1)) - 1
        rets.fillna(0, inplace=True)
        rets.replace([np.inf, -np.inf], 0, inplace=True)

        return df, rets, ann_sharpes_list, sharpes_list, number_trades, df_trades_results

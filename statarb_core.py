# import from binance_market_module import BinanceMarketModule
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import adfuller
import scipy.stats as st
pd.options.mode.chained_assignment = None  # default='warn'

# create class to generate signal, statarbV3
# implement signals aggregator
class StatArb:
    
    # constructor
    def __init__(self, df: pd.DataFrame, interval='day'):
        # initialise variables
        self.df = df
        self.position_state = 0
        self.position_states = []
        self.transactions = 0
        self.transactions_count = []
        self.transaction_happen = 0
        self.transaction_happen_records = []
        self.transaction_cost = []
        self.transaction_cost_param = 0
        self.annual_factor = 365
        self.momentum_halflife = 7 # 1-week duration for half life
        self.reversion_halflife = 14
        self.tickers = df.columns
        
        # determine interval
        if interval == 'day':
            self.annual_factor = 365
        elif interval == 'hour':
            self.annual_factor = 365 * 24
        else:
            self.annual_factor = 365
            
        # check for required number of assets
        if df.shape[1] == 2:
            data = df.copy()
            # generate relevant figures for statistical arbitrage
            assets = list(data.columns)
            rename = {k: v for k, v in zip(assets, ['x', 'y'])}
            data.rename(columns = rename, inplace=True)
            for symbol in list(data.columns):
                data['norm_' + symbol] = data[symbol] / data[symbol][0]
                data['pct_change_' + symbol] = data[symbol].pct_change()
                
            # generate price ratio x/y
            symbols = list(data.columns)
            
            # ratio = x / y
            # if ratio is high, this means we want to short X, and long Y
            # if ratio is low, we want to long X and short Y
            data['ratio'] = data['norm_' + symbols[0]] / data['norm_' + symbols[1]]
            self.data = data
            self.symbols = data
        else:
            print('Module requires only 2 assets')
            
    # reset class
    def reset(self):
        self.__init__(self.df)
        self.p_data = None
        
    # generate signal based on rolling window, mean reversion
    def generate_signal(self, confidence_level: float, weights: list =[0.5, 0.5]):
        self.confidence_level = confidence_level
        
        # process data, respecting the window
        self.p_data = self.data.copy()
        self.p_data['ratio_mean'] = self.p_data['ratio'].ewm(halflife=self.reversion_halflife).mean()
        self.p_data['ratio_std'] = self.p_data['ratio'].ewm(halflife=self.reversion_halflife).std()
        
        # generate signal using internal static function; and apply
        self.p_data['reversion_signal'] = self.generate_reversion_signal()['reversion_signal'] * weights[0]
        self.p_data['momentum_signal'] = self.generate_momentum_signal()['momentum_signal'] * weights[1]
        self.p_data['signal_strength'] = self.aggregate_signals(weights)
        self.p_data.dropna(how='any', axis=0, inplace=True)
        return self.p_data
    
    # based on signal strength, obtain positions
    # only one position at a time
    def generate_positions(self, transaction_cost: float = 0):
        # iterate through processed data, and generate positions
        # position state = 1 --> short X, long Y
        # position state = -1 --> long X, short Y
        
        self.transaction_cost = transaction_cost
        
        for i in range(len(self.p_data)):

            # extract data from a row
            values = self.p_data.iloc[i, :]
            signal_strength = values['signal_strength']
            ratio_mean = values['ratio_mean']
            ratio = values['ratio']
            self.transaction_happen = 0

            # if there is no current position, check for signal strength
            if self.position_state == 0:
                if signal_strength > 0:
                    self.position_state = 1
                    self.transactions += 1
                    self.transaction_happen = 1
                elif signal_strength < 0:
                    self.position_state = -1
                    self.transactions += 1
                    self.transaction_happen = 1
                else:
                    self.position_state = 0
                    
            # if a position already exists, check whether we can close it
            elif self.position_state == 1:

                # if ratio has fallen below the mean, we can close the position
                if ratio < ratio_mean:
                    self.position_state = 0 # close position
                    self.transactions += 1
                    self.transaction_happen = 1
                # otherwise, maintain the position
                else:
                    self.position_state = 1 # maintain position
                    self.transaction_happen = 0                    

            # if a position already exists, check whether we can close it
            elif self.position_state == -1:
                if ratio > ratio_mean:
                    self.position_state = 0 # close position
                    self.transactions += 1
                    self.transaction_happen = 1
                else:
                    self.position_state = -1 # maintain position
                    self.transaction_happen = 0
            else:
                print('Position State should only be 1 or -1')

            # store position states and transaction costs
            self.position_states.append(self.position_state)
            self.transactions_count.append(self.transactions)
            self.transaction_happen_records.append(self.transaction_happen)
        
        # append to processed data
        period_shift = 1
        self.p_data['position_state'] = self.position_states
        self.p_data['adj_position_state'] = self.p_data['position_state'].shift(periods=period_shift) # should shift by 1
        self.p_data['transaction_cost'] = self.transaction_happen_records
        self.p_data['transaction_cost'] = self.p_data['transaction_cost'].apply(lambda x: x * self.transaction_cost)
        self.p_data['adj_transaction_cost'] = self.p_data['transaction_cost'].shift(periods=period_shift) # should shift by 1
        self.p_data['cumulative_transactions'] = self.transactions_count
        
        # compute period returns
        # position state = 1 --> short X, long Y: 
        # position state = -1 --> long X, short Y
        # deduct transaction costs
        self.p_data['period_return'] = - (self.p_data['adj_position_state'] * self.p_data['pct_change_x']) \
                                       + (self.p_data['adj_position_state'] * self.p_data['pct_change_y']) \
                                       - self.p_data['adj_transaction_cost']
        
        # compute cumulative returns
        self.p_data['cum_return'] = (self.p_data['period_return'] + 1).cumprod() - 1
        
    # obtain net position
    def generate_net_portfolio(self):
        # positive if overall long, negative if overall short        
        net_portfolio = 0
        long_portfolio = 1
        short_portfolio = 1
        all_net_portfolio = []
        for i in range(len(self.p_data)):
            values = self.p_data.iloc[i, :]
            position_state = values['adj_position_state']
            if position_state == 0:
                net_portfolio = 0  # all cash, no position taken
                # reset long and short portfolios
                long_portfolio = 1
                short_portfolio = 1
                
                # position_state = 1: short X, long Y
                # position_state = -1: short Y, long X
                # net position is out of entire position
                # store long and shoort
            elif position_state == 1:
                long_portfolio = (long_portfolio) * (1 + values['pct_change_y'])
                short_portfolio = (short_portfolio) * (1 + values['pct_change_x'])
                net_portfolio = (long_portfolio - short_portfolio) / (abs(long_portfolio) + abs(short_portfolio))
            else:
                long_portfolio = (long_portfolio) * (1 + values['pct_change_x'])
                short_portfolio = (short_portfolio) * (1 + values['pct_change_y'])
                net_portfolio = (long_portfolio - short_portfolio) / (abs(long_portfolio) + abs(short_portfolio))
                            
            # store net position
            all_net_portfolio.append(net_portfolio)

        # update net position
        self.p_data['net_portfolio'] = all_net_portfolio
        
    # plot performance   
    def plot_all(self, fg: tuple = (20, 2)):
        z_score = StatArb.get_z_score(self.confidence_level)
        d = self.p_data.copy()

        plt.figure(figsize=fg)
        plt.plot(d[['cum_return']])
        plt.title('Cumulative Returns')
        
        plt.figure(figsize=fg)
        plt.plot(d['ratio'])
        plt.plot(d['ratio_mean'])
        plt.fill_between(d.index, d['ratio_mean'] - z_score * d['ratio_std'], d['ratio_mean'] + z_score * d['ratio_std'], color='lightblue')
        plt.title('Ratio Time Series, with Confidence Interval')

        # plt.figure(figsize=fg)
        # plt.plot(d['norm_x'])
        # plt.plot(d['norm_y'])
        # plt.title('Normalised X and Y')

        plt.figure(figsize=fg)
        plt.plot(d['reversion_signal'])
        plt.plot(d['momentum_signal'])
        plt.legend(['Mean Reversion', 'Momentum'])
        plt.title('Individual Signals')

        plt.figure(figsize=fg)
        plt.plot(d['signal_strength'])
        plt.title('Signal Strength')

        plt.figure(figsize=fg)
        plt.plot(d[['position_state']])
        plt.title('Position States')

        # plt.figure(figsize=fg)
        # plt.plot(d['cumulative_transactions'])
        # plt.title('Cumulative Transactions')

        plt.figure(figsize=fg)
        plt.plot(d['net_portfolio'])
        plt.title('Net Long/Short Portfolio Position')
        plt.show()
        
    # generate results summary
    def results(self):
        # generate results for portfolio, as well as individual assets
        self.p_data.dropna(how='any', axis=0, inplace=True)

        sharpe = np.mean(self.p_data['period_return']) / np.std(self.p_data['period_return']) * np.sqrt(self.annual_factor)
        sortino = np.mean(self.p_data['period_return']) / np.std(self.p_data[self.p_data['period_return'] < 0]['period_return']) * np.sqrt(self.annual_factor)
        max_drawdown = StatArb.max_drawdown(self.p_data.dropna()['cum_return'] + 1)
        cumulative_return = self.p_data['cum_return'].values[-1]
        cagr = (self.p_data['cum_return'].values[-1] + 1) ** (365 / len(self.p_data)) - 1
        
        # constructing output from constituent assets
        sharpe_x = np.mean(self.p_data['pct_change_x']) / np.std(self.p_data['pct_change_x']) * np.sqrt(self.annual_factor)
        sortino_x = np.mean(self.p_data['pct_change_x']) / np.std(self.p_data[self.p_data['pct_change_x'] < 0]['pct_change_x']) * np.sqrt(self.annual_factor)
        max_drawdown_x = StatArb.max_drawdown(self.p_data.dropna()['norm_x'] + 1)
        cumulative_return_x = (self.p_data['x'].values[-1] - self.p_data['x'].values[0]) / self.p_data['x'].values[0]
        cagr_x = (self.p_data['x'].values[-1] / self.p_data['x'].values[0]) ** (365/len(self.p_data['x'].values)) - 1
        
        sharpe_y = np.mean(self.p_data['pct_change_y']) / np.std(self.p_data['pct_change_y']) * np.sqrt(self.annual_factor)
        sortino_y = np.mean(self.p_data['pct_change_y']) / np.std(self.p_data[self.p_data['pct_change_y'] < 0]['pct_change_y']) * np.sqrt(self.annual_factor)
        max_drawdown_y = StatArb.max_drawdown(self.p_data.dropna()['norm_y'] + 1)        
        cumulative_return_y = (self.p_data['y'][-1] - self.p_data['y'][0]) / self.p_data['y'][0]
        cagr_y = (self.p_data['y'].values[-1] / self.p_data['y'].values[0]) ** (365/len(self.p_data['y'].values)) - 1
        
        # constructing output from portfolio
        # dic['confidence_level'] = self.confidence_level
        # dic['window'] = int(self.window)
        # dic['transaction_cost'] = self.transaction_cost
        output = pd.DataFrame(index=['Sharpe', 'Sortino', 'Max Drawdown', 'CAGR', 'Cumulative Return'])
        
        # from portfolio
        output['Portfolio'] = [sharpe, sortino, max_drawdown, cagr, cumulative_return]
        output[self.tickers[0]] = [sharpe_x, sortino_x, max_drawdown_x, cagr_x, cumulative_return_x]
        output[self.tickers[1]] = [sharpe_y, sortino_y, max_drawdown_y, cagr_y, cumulative_return_y]
        
        return output
        
        
    # generate z_score
    @staticmethod
    def get_z_score(confidence_level: float):
        confidence_input = 1 - (1-confidence_level)/2     # bounding to 0.999
        z_score = st.norm.ppf(confidence_input)
        return z_score


    # generate mean reversion signal, providing confidence interval
    def generate_reversion_signal(self):
        testbed = self.data.copy()
        # testbed['ratio_mean'] = testbed['ratio'].rolling(self.window).mean()
        testbed['ratio_mean'] = testbed['ratio'].ewm(halflife=self.reversion_halflife).mean()
        testbed['ratio_std'] = testbed['ratio'].ewm(halflife=self.reversion_halflife).std()
        testbed['reversion_signal'] = testbed.apply(
            lambda row: StatArb.get_single_reversion_signal(row['ratio'],
                                                row['ratio_mean'],
                                                row['ratio_std'],
                                                self.confidence_level), axis=1)
        return testbed
       
    # generate momentum signal, providing half-life
    def generate_momentum_signal(self, hl=None):
        # check if halflife is specified, otherwise use default
        if not hl:
            hl = self.momentum_halflife

        # generate signals from base table
        momentum_threshold = 0.000
        testbed = self.data.copy()
        testbed['ratio_pct_change'] = testbed['ratio'].pct_change()
        testbed['ratio_change_std'] = testbed['ratio_pct_change'].ewm(halflife=hl).std()
        testbed['raw_momentum_signal'] = testbed['ratio_pct_change'].ewm(halflife=hl).mean() * (-1)
        testbed['momentum_signal'] = testbed['raw_momentum_signal'].apply(lambda x: x if abs(x) > momentum_threshold else 0)
        return testbed

    
    # create function to aggregate both mean reversion and momentum signals
    def aggregate_signals(self, weights: list()):
        # generate weighted combination of signals
        # mean_reversion : momentum
        reversion_signal = self.generate_reversion_signal()['reversion_signal']
        momentum_signal = self.generate_momentum_signal()['momentum_signal']
        overall_signal = weights[0] * reversion_signal + weights[1] * momentum_signal
        return overall_signal
        
# generate mean reversion signal
    @staticmethod
    def get_single_reversion_signal(ratio, ratio_mean, ratio_std, confidence_level):
        z_score = StatArb.get_z_score(confidence_level)

        # generate lower and upper bounds
        lower_bound = ratio_mean - (z_score * ratio_std)
        upper_bound = ratio_mean + (z_score * ratio_std)

        # generate signal based on lower and upper bound
        if ratio > upper_bound:
            #signal_strength = (ratio - upper_bound) / upper_bound
            signal_strength = (ratio - upper_bound) / ratio
        elif ratio < lower_bound:
            #signal_strength = (ratio - lower_bound) / lower_bound
            signal_strength = (ratio - lower_bound) / ratio
        else:
            signal_strength = 0
        return signal_strength        
    
    @staticmethod
    def max_drawdown(prices):
        # solving using dynamic programming, where we can achieve linear time complexity
        # initiatise max drawdown and running peak
        all_max_drawdowns = []
        all_drawdowns = []
        max_drawdown = 0
        running_peak = 0

        for price in prices:
            # obtain new peak
            running_peak = max(running_peak, price)

            # obtain new drawdown and calculate new max drawdown
            drawdown = (running_peak - price) / running_peak
            max_drawdown = max(max_drawdown, drawdown)
            all_max_drawdowns.append(max_drawdown)
            all_drawdowns.append(drawdown)

        return max_drawdown

    
    def plot_ratio(self):
        if self.data:
            self.data[['ratio']].plot(figsize=(20, 5))
            plt.title('Price Ratio')
            plt.ylabel('Normalised X / Normalised Y')
            plt.show()

            

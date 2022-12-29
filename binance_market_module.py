# Binance Market Module for Data Storage and Retrieval of spot, linear futures and inverse futures klines data

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import datetime as dt
from dateutil.relativedelta import relativedelta
import math
import os
import requests  # requests module to interact with endpoints

class BinanceMarketModule:
    
    # provide class variables
    spot_base_url = 'https://api3.binance.com'
    lin_futures_base_url = 'https://fapi.binance.com'
    inv_futures_base_url = 'https://dapi.binance.com'
    
    # store symbolmaster paths
    spot_smpath = 'spot_symbolmaster.feather'
    lin_futures_smpath = 'lin_symbolmaster.feather'
    inv_futures_smpath = 'inv_symbolmaster.feather'
    
    # wait time
    wait = 0.1
    
    # intervals: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
    
    # initialise constructor to token_type input
    def __init__(self, token_type: str):
        self.token_type = token_type
        self._assign_api()  # assign the right api
        self.base_url = self._assign_api()
        self.smpath = self._smpath()
        self.symbolmaster = None
        self.tokenlist = None
        self.parent_dir = os.getcwd()
        self.data_dir = os.path.join(self.parent_dir, 'data')
        
        # construct empty data directory if it does not exist
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        
    # update symbolmaster
    def pullSymbolMaster(self) -> pd.DataFrame:
        # check if symbolmaster already exists in local directory
        self.smpath= self._smpath()
        if os.path.exists(self.smpath):
            df = pd.read_feather(self.smpath)

        # if symbolmaster does not currently exist, construct new symbolmaster
        else:
            exchange_url = '/api/v3/exchangeInfo'
            endpoint = self.base_url + exchange_url
            response = self._get(endpoint)
            symbolmaster_cols = ['symbol', 'status', 'baseAsset', 'quoteAsset']
            df = pd.DataFrame(response['symbols'])[symbolmaster_cols]
            df['timestamp_utc'] = dt.datetime.now(dt.timezone.utc)
            df['date_utc'] = df['timestamp_utc'].apply(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"))
        
        self.symbolmaster = df
        self.tokenlist = df['symbol'].values
        return self.symbolmaster
    
    # store symbolmaster as feather
    def storeSymbolMaster(self):
        # if it currently does not exist
        if self.symbolmaster is None:
            self.pullSymbolMaster()
        # if symbolmaster exists
        else:
            self.symbolmaster.to_feather(self.smpath)
            print(f'Symbol Master stored as {self.smpath}')
        return self.symbolmaster
    
    # single pull ticker data
    def pullOHLCV(self, symbol: str, interval: str, start: str, end: str)-> pd.DataFrame:
        # manage duration; start and end formatted as string
        unix_start = self.string_to_unixtime(start)
        unix_end = self.string_to_unixtime(end)
        kline_url = '/api/v3/klines'
        full_url = self.base_url + kline_url
        params = {'symbol': symbol,
                  'interval': interval,
                  'limit': 1000,
                  'startTime': unix_start,
                  'endTime': unix_end
                 }
        result = self._get(full_url, params=params)
        
        # check for result
        if len(result) == 0:  # no output returned from API call
            df = None
        else:            
            cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'trades']
            df = pd.DataFrame(result).iloc[:, :7]
            df.columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time']
            df['open_ts'] = df['open_time'].apply(lambda x: dt.datetime.fromtimestamp(x/1000))
            df['close_ts'] = df['close_time'].apply(lambda x: dt.datetime.fromtimestamp(x/1000))
            df['symbol'] = symbol
            df = df[['symbol', 'open_ts', 'open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                    'close_ts']]

            # convert prices to float
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
        return df
    
    # pull klines by date, with pagination
    def pullBulkOHLCV(self, symbol: str, interval: str, start: str, end: str) -> pd.DataFrame:
        # initiate bulk pull
        bulk_list = []
        new_start = start  # comparison variable
        
        while new_start < end:
            # check on first data pull to see if it exists
            df = self.pullOHLCV(symbol, interval, new_start, end)
            if df is not None:
                bulk_list.append(df)
                
                # using extracted data to determine next start
                first_close = df['close_ts'].iloc[0].strftime('%Y-%m-%d %H:%M:%S')  # convert to string format
                last_close = df['close_ts'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')
                last_open = df['open_ts'].iloc[-1].strftime('%Y-%m-%d %H:%M:%S')

                # update new start to last close
                new_start = last_close

                # exit loop if exceeded intended last time
                if new_start > end:
                    break
            # if no data output from api call, we iterate next start by a week, forsaking some data at the start
            else:
                new_start = (dt.datetime.strptime(new_start, '%Y-%m-%d %H:%M:%S')
                             +relativedelta(months=1)).strftime('%Y-%m-01 %H:%M:%S')  # update new start to first day of next month
                
        # merge dataframes if bulk_list exists
        if len(bulk_list) > 0:
            agg_df = pd.concat(bulk_list, axis=0) # merge all dataframes together
            agg_df.reset_index(drop=True, inplace=True) # reset index
            agg_df.drop(agg_df.index[-1], inplace=True) # exclude last row
        else:
            agg_df = None
        return agg_df
    
    # store bulk OHLCV into local directory, for any duration
    def storeBulkOHLCV(self, symbol: str, interval: str, start_date: str, end_date: str) -> str:
        # after constructing dataframe with pagination, store into local directory
        start_timestring = self.datestring_to_timestring(start_date)
        end_timestring = self.datestring_to_timestring(end_date)
        df = self.pullBulkOHLCV(symbol, interval, start_timestring, end_timestring)
        
        # save in appropriate format
        if df is not None:
            ohlcv_path = symbol + '_' + interval + '_' + start_date + '_' + end_date
            df.to_feather(os.path.join(self.data_dir, ohlcv_path))
            print(f'New File {ohlcv_path} constructed in DataStore')
            first_open = df['open_ts'].iloc[0]
            last_close = df['close_ts'].iloc[-1]
            #print(f'{symbol} for {interval} interval stored for duration {first_open} to {last_close}')
            return df
        else:
            # inform if no data available for that time
            print(f'No data available for duration {start_date} to {end_date}')
        
    # update data store and arrange files into library
    def singleUpdateDataStore(self, symbol: str, interval: str, start_date: str, end_date: str):
        # construct sequence of all date breaks, correcting start date to be always first day of month
        
        # determine step based on interval
        date_step, step_format = self._manage_library(interval)
        
        # if start date is not first day of month, then change start date
        if int(dt.datetime.strptime(start_date, '%Y-%m-%d').strftime('%d')) != 1:            
            start_date = dt.datetime.strptime(dt.datetime.strptime(start_date, '%Y-%m-%d').strftime(step_format), step_format).strftime('%Y-%m-%d')
        
        # if end date is not first day of month, then change end date
        if int(dt.datetime.strptime(end_date, '%Y-%m-%d').strftime('%d')) != 1:
            end_date = (dt.datetime.strptime(dt.datetime.strptime(end_date, '%Y-%m-%d').strftime(step_format), step_format) + relativedelta(months=date_step)).strftime('%Y-%m-%d')
                
        # start constructing sequence
        date_sequence = []
        while start_date < end_date:
            date_sequence.append(start_date)
            start_date = (dt.datetime.strptime(start_date, '%Y-%m-%d') + relativedelta(months=date_step)).strftime('%Y-%m-%d')

        # use date sequence to store files
        for i in range(len(date_sequence) - 1):
            start = date_sequence[i]
            end = date_sequence[i+1]
            filepath = symbol + '_' + interval + '_' + start + '_' + end
            full_filepath = os.path.join(self.data_dir, filepath)
            
            # check if file already exists
            if os.path.exists(full_filepath):
                print(f'File {filepath} already exists in DataStore')
                continue
                
            # if file currently does not exist, then construct new file
            else:
                self.storeBulkOHLCV(symbol, interval, start, end)
                
                # implement wait time to avoid hitting api rate limit
                time.sleep(self.wait)
    
    # retrieve and reconstruct from local directory
    def retrieveOHLCV(self, symbol: str, interval: str, start_date: str, end_date: str):
        # determine step based on interval
        date_step, step_format = self._manage_library(interval)
        
        # construct file paths for retrieval, with addition of latest
        filepaths = self._get_filepaths_retrieval(symbol, interval, start_date, end_date)
        
        # loop through and construct aggregated dataframe
        df_list = []
        for fp in filepaths:
            full_fp = os.path.join(self.data_dir, fp)
            if os.path.exists(full_fp):
                df = pd.read_feather(full_fp)
                df_list.append(df)
            else:
                continue
        
        # merge all dataframes and reset index
        overall_df = pd.concat(df_list, axis=0)
        overall_df.reset_index(drop=True, inplace=True)
        
        # drop excess data outside of selected range
        output = overall_df[(overall_df['open_ts'] < end_date) & (overall_df['open_ts'] >= start_date)]
        return output
    
    # assign appropriate api_urls
    def _assign_api(self) -> str:
        if self.token_type == 'spot':
            self.base_url = self.spot_base_url
        elif self.token_type == 'lin_future':
            self.base_url = self.lin_futures_base_url
        elif self.token_type == 'inv_future':
            self.base_url = self.inv_futures_base_url
        else:
            self.base_url = None
            print('Unknown Token Type')
        return self.base_url
            
    # get request for usage within the class
    def _get(self, full_endpoint: str, params: dict = None):
        r = requests.get(full_endpoint, params=params)
        results = r.json()
        return results
    
    def _smpath(self) -> str:
        if self.token_type == 'spot':
            self.smpath = self.spot_smpath
        elif self.token_type == 'lin_future':
            self.smpath = self.lin_futures_smpath
        elif self.token_type == 'inv_future':
            self.smpath = self.inv_futures_smpath
        else:
            self.smpath = None
            print('Unknown Token Type')
        return self.smpath
    
    # library management heuristics
    def _manage_library(self, interval: str):
        # implement simple library management heuristics, to return package size
        if interval == '1m':
            package_size = 1  # months
            step_format = '%Y-%m'
        elif interval == '1h':
            package_size = 1 # months
            step_format = '%Y-%m'
        elif interval == '1d':
            package_size = 12 # months
            step_format = '%Y'
        else:
            print('Choose appropriate interval - 1m, 1h or 1d')
            package_size = None
        return package_size, step_format
    
    # construct file date sequence list, ensure files collected inclusive of all in selected date range
    def _get_filepaths_retrieval(self, symbol: str, interval:str, start_date: str, end_date: str) -> list:
        files = []
        
        # determine step based on interval
        date_step, step_format = self._manage_library(interval)
        
        # start constructing date sequence
        date_sequence = []
        
        # select one date step beyond range, normalise file start to start of month
        file_start = dt.datetime.strptime(start_date, '%Y-%m-%d').strftime('%Y-%m-01')
        
        # while start_date < end_date:
        while dt.datetime.strptime(file_start, '%Y-%m-%d') < dt.datetime.strptime(end_date, '%Y-%m-%d') + relativedelta(months=date_step):
            date_sequence.append(file_start)
            file_start = (dt.datetime.strptime(file_start, '%Y-%m-%d') + relativedelta(months=date_step)).strftime('%Y-%m-%d')

        # construct filepaths
        for i in range(len(date_sequence) - 1):
            start = date_sequence[i]
            end = date_sequence[i+1]
            filepath = symbol + '_' + interval + '_' + start + '_' + end
            files.append(filepath)
                        
        return files
    
    # create methods for handling time conversions
    @staticmethod
    def string_to_unixtime(timestring: str):  # convert this into UTC unixtime
        unixtime = time.mktime(dt.datetime.strptime(timestring, '%Y-%m-%d %H:%M:%S').timetuple()) * 1000
        return int(unixtime)
    
    @staticmethod
    def datestring_to_timestring(datestring: str):  # convert this into UTC unixtime
        ts = dt.datetime.strptime(datestring, '%Y-%m-%d')
        timestring = ts.strftime('%Y-%m-%d %H:%M:%S')
        return timestring
    
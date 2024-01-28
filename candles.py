import yfinance as yf
import pandas as pd
import numpy as np
import datetime as dt
import os
import matplotlib.pyplot as plt
from typing import List, Optional
# from scipy.optimize import minimize

class Candles:

    """
    1. Download candles using yfinance and store locally
    2. Read, clean and output candles dataframe for single or multiple tickers
    """

    # construct candles class to extract candles from any subset of tickers
    def __init__(self, sec_master_filepath: str, candles_folder: str, interval: str):

        # store local variables
        self.current_dir = os.getcwd()
        self.sm_filepath = sec_master_filepath
        self.sm = pd.read_parquet(sec_master_filepath)
        self.candles_folder = os.path.join(self.current_dir, candles_folder)
        self.all_tickers = list(self.sm['symbol'].values)
        self.data_dic = None
        self.interval = interval

    # update state of data when reading
    def update_data_state(self, force_update: bool = False):

        # if data dic already exists or if I do not want to force update
        if self.data_dic and force_update == False:
            return self.data_dic
        else:
            self.data_dic = {}

        # retrieve filepaths
        datastore_filepaths = [os.path.join(self.candles_folder, x) for x in os.listdir(self.candles_folder)] # get all filenames and extract tickers

        # iterate along all filepaths
        for f in datastore_filepaths:
            ticker = f.split('/')[-1].split('_')[0] # extracting ticker from filepath
            df = pd.read_parquet(f)

            # store min and max dates of data
            min_date = min(df.index).strftime('%Y-%m-%d')
            max_date = max(df.index).strftime('%Y-%m-%d')
            self.data_dic[ticker] = (min_date, max_date)

        # compile at the end
        print('Data Dictionary Updated')
        return self.data_dic


    def load_local_ticker_file(self, ticker: str):
        ticker_filepath = os.path.join(self.candles_folder, f"{ticker}_{self.interval}.pq")
        ticker_df = pd.read_parquet(ticker_filepath)

        # add additional features
        ticker_df['close_abs_change'] = ticker_df['Close'].diff()
        ticker_df['close_pct_change'] = ticker_df['Close'].pct_change()
        ticker_df['ticker'] = ticker # store ticker name as well

        # return ticker df
        return ticker_df
    
    # bulk load multiple tickers
    def load_candles(self, ticker_list: List[str] = None, date_from: str = None, date_to: str = None) -> pd.DataFrame:

        # if ticker list not provided, use all tickers instead
        if not ticker_list:
            ticker_list = self.all_tickers

        # for each ticker, retrieve candles
        output_df_list = []
        for ticker in ticker_list:

            try:
                df = self.load_local_ticker_file(ticker)

                if len(df) > 0:
                    # checking daterange
                    if not date_from:
                        date_from = min(df.index).strftime('%Y-%m-%d')

                    if not date_to:
                        date_to = max(df.index).strftime('%Y-%m-%d')

                    output_df = df[(df.index >= date_from) & (df.index <= date_to)] # filter based on dates
                    output_df_list.append(output_df)

                # continue to next loop  
                else:
                    pass
            except Exception as e:
                print(e)
        
        # check on combined output
        if len(output_df_list) > 0: 
            combined_df = pd.concat(output_df_list, axis=0)
            return combined_df
        else:
            return None

    # update local candles files for selected tickers, period always max
    def update_candles(self, selected_tickers: List[str]):

        # iterate along all selected tickers
        for i, ticker in enumerate(selected_tickers):

            # construct filename, extract data and save file
            try:
                # save to local candles folder
                filepath = os.path.join(self.candles_folder, f"{ticker}_{self.interval}.pq")
                ticker_obj = yf.Ticker(ticker)

                # perform extraction and update data dic
                df = ticker_obj.history(period='max', interval=self.interval)
                
                # save dataframe as parquet file only if not empty
                if len(df) > 0:
                    df.to_parquet(filepath) # store file locally
                    min_date = min(df.index).strftime('%Y-%m-%d') # get min date
                    max_date = max(df.index).strftime('%Y-%m-%d') # get max date
                    self.data_dic[ticker] = (min_date, max_date) # update data dic
                    print(f"Data Dictionary Updated for {ticker}: ({min_date, max_date})")
                else:
                    print(f"No data retrieved for ticker: {ticker}, no file stored")
                    pass

            except Exception as e:
                print(f"{i}: {ticker}: {e}")
                pass

            # checking on iteration
            if i % 100 == 0:
                print(f"{i} ticker candles stored/updated so far.")
from binance_market_module import BinanceMarketModule
import time
import datetime as dt
from dateutil.relativedelta import relativedelta

if __name__ == "__main__":
    
    update_start_time = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f'Initialising Data Store Update at {update_start_time}')
    
    # initialise module instance and select time duration
    bmod = BinanceMarketModule('spot')
    start_date = '2017-01-01'
    end_date = '2022-12-01'

    # check latest symbol master and retrieve usdt tokens
    sm = bmod.pullSymbolMaster()
    selected_base_tokens = ['BTC', 'ETH', 'LTC', 'ADA', 'XRP', 'AAVE']
    selected_symbols = sm[(sm['quoteAsset'] == 'USDT') & (sm['baseAsset'].isin(selected_base_tokens))]['symbol'].tolist()
    
    
    # run for daily
    interval = '1d'
    for token in selected_symbols:
        print(f'Initiating Pull for {token}...')
        time.sleep(3)  # implement sleep to prevent hitting api rate limit
        bmod.singleUpdateDataStore(token, interval, start_date, end_date)
    print(f'Completed Datastore Update for data - Daily')
        
    # run for hourly
    interval = '1h'
    for token in selected_symbols:
        print(f'Initiating Pull for {token}...')
        time.sleep(3)  # implement sleep to prevent hitting api rate limit
        bmod.singleUpdateDataStore(token, interval, start_date, end_date)
    print(f'Completed Datastore Update for data - Hourly')
    
    # run for minute bars
    interval = '1m'
    for token in selected_symbols:
        print(f'Initiating Pull for {token}...')
        time.sleep(3)  # implement sleep to prevent hitting api rate limit
        bmod.singleUpdateDataStore(token, interval, start_date, end_date)
    print(f'Completed Datastore Update for data - Minute Bars')
    
    # final update
    update_end_time = dt.datetime.now()
    print(f'Completed Data Store Update at {update_end_time.strftime('%Y-%m-%d %H:%M:%S')}')
    duration_minutes = (update_end_time - update_start_time).total_seconds() / 60
    print(f'Update duration is {duration_minutes}')
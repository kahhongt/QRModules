from binance_market_module import BinanceMarketModule
import datetime as dt
import time

# initialise module instance and select time duration
bmod = BinanceMarketModule('spot')
start_date = '2017-01-01'
end_date = dt.datetime.now().strftime('%Y-%m-%d')
interval = '1h'

# check latest symbol master and retrieve usdt tokens
sm = bmod.pullSymbolMaster()
selected_base_tokens = ['BTC', 'ETH', 'LTC', 'ADA', 'XRP', 'AAVE']
selected_symbols = sm[(sm['quoteAsset'] == 'USDT') & (sm['baseAsset'].isin(selected_base_tokens))]['symbol'].tolist()

# check symbolmaster and ingest data
for token in selected_symbols:
    # check if it currently exists:
        
    print(f'Initiating Pull for {token}...')
    time.sleep(3)  # implement sleep to prevent hitting api rate limit
    bmod.singleUpdateDataStore(token, interval, start_date, end_date)
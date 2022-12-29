from binance_market_module import BinanceMarketModule
import time

# initialise module instance and select time duration
bmod = BinanceMarketModule('spot')
start_date = '2017-01-01'
end_date = '2022-12-01'
interval = '1d'

# check latest symbol master and retrieve usdt tokens
sm = bmod.pullSymbolMaster()
usdt_tokens = sm[sm['quoteAsset'] == 'USDT']['symbol'].tolist()

# check symbolmaster and ingest data
for token in usdt_tokens:
    print(f'Initiating Pull for {token}...')
    time.sleep(1)  # implement sleep to prevent hitting api rate limit
    bmod.singleUpdateDataStore(token, interval, start_date, end_date)
# QR Modules
Develop modules in Python to perform essential signal processing tasks after ingesting klines data of any asset, with the objective of generating, storing and executing alpha signals.

**Binance Market Module**: (DONE)
1. Extract relevant candlesticks data from Binance API - Spot (Done), Perpetual and Delivery Futures
2. Apply pagination to account for API rate limits (Done)
3. Store data as flat files in local drive (Done - automatically creates data folder to store files locally)
4. Update local drive on each run, after checking and storing state of local drives

**Forex and Futures Market Module**:
1. Extract relevant candlesticks data from IBKR and OpenBB
3. Apply pagination to account for API rate limits
4. Store data as flat files in local drive
5. Update local drive on each run

**Equities Module**:
1. Equities Tickers Sec Master extracted using YFinance (Done)
2. Bulk upload Candlesticks Data across selected tickers, over selected intervals (Done)

**Analysis Module**:
1. Quick Visualisations of Prices based on Symbol Input,
2. Implementing Smoothing Functions --> Moving Average, Differencing, Holt-Winters Exponential Smoothing
3. Linear Regression Analysis
4. Autocorrelation and Cross-correlation Calculations
5. Principal Component Analysis
6. Wavelet Analysis --> decompose a signal into component frequencies
7. Seasonal Decomposition
8. Box Cox Transformation --> transform a time series into a more normal distribution
9. Potential Alpha Signal Recognition

**Signals Module**:
1. Manage Signals Generation
2. Backtest Engine: Ingest signals and backtest portfolio
3. Returns Analysis: Annualised, Sharpe, Sortino, etc

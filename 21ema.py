import numpy as np
import pandas as pd
import yfinance as yf
from scipy.signal import find_peaks
from backtesting import Strategy, Backtest
from backtesting.lib import crossover


# identify swing highs and lows
def find_swing_highs_lows(data, lookback=10):
    """Identifies swing highs and lows in the price data."""
    highs, _ = find_peaks(data['High'], distance=lookback)  # Detect swing highs
    lows, _ = find_peaks(-data['Low'], distance=lookback)   # Detect swing lows (invert price for lows)
    return highs, lows


# calculate ema
def EMA(series, period):
    """Exponential Moving Average (EMA) Calculation"""
    return pd.Series(series).ewm(span=period, adjust=False).mean()


# calculate adx
def ADX(data, period=14):
    """calculates ADX to determine if the market is trending"""
    df = data.copy()
    df['TR'] = np.maximum(df['High'] - df['Low'], 
                          np.maximum(abs(df['High'] - df['Close'].shift(1)), abs(df['Low'] - df['Close'].shift(1))))
    
    df['DM+'] = np.where((df['High'] - df['High'].shift(1)) > (df['Low'].shift(1) - df['Low']), 
                         np.maximum(df['High'] - df['High'].shift(1), 0), 0)
    df['DM-'] = np.where((df['Low'].shift(1) - df['Low']) > (df['High'] - df['High'].shift(1)), 
                         np.maximum(df['Low'].shift(1) - df['Low'], 0), 0)

    df['TR'] = df['TR'].rolling(period).sum()
    df['DM+'] = df['DM+'].rolling(period).sum()
    df['DM-'] = df['DM-'].rolling(period).sum()

    df['DI+'] = 100 * (df['DM+'] / df['TR'])
    df['DI-'] = 100 * (df['DM-'] / df['TR'])
    df['DX'] = 100 * (abs(df['DI+'] - df['DI-']) / (df['DI+'] + df['DI-']))
    df['ADX'] = df['DX'].rolling(period).mean()
    
    return df['ADX']


# position size calculator
def calculate_position_size(balance, entry, stop_loss, risk_percent):
    """ensures exactly 1% risk per trade"""
    
    risk_amount = balance * (risk_percent / 100)  # 1% risk 
    risk_distance = abs(entry - stop_loss)  # distance between entry and SL
    position_size = risk_amount / risk_distance  # units based on risk

    # ensures size is valid: Whole number if >=1, fraction of equity if <1
    if position_size >= 1:
        return round(position_size)  
    else:
        return position_size / balance  



class EURUSD_TradingStrategy(Strategy):
    ema_period = 21
    adx_period = 14  # ADX period to filter trends
    adx_threshold = 25  # only enter if ADX > 25
    lookback_swing = 10  # swing point detection
    risk_percent = 1  


    def init(self):
        """Initialize the 21 EMA & ADX indicators"""
        self.ema21 = self.I(EMA, self.data.Close, self.ema_period)  # 21 EMA
        self.adx = self.I(ADX, self.data.df, self.adx_period)  # ADX for trend strength

    def next(self):
        """executes trading logic on each new candlestick"""
        price = self.data.Close[-1]
        ema_value = self.ema21[-1]
        adx_value = self.adx[-1]

        # find swing highs and lows
        highs, lows = find_swing_highs_lows(self.data.df, lookback=self.lookback_swing)

        if len(highs) < 1 or len(lows) < 1:
            return

        swing_high = self.data.High[highs[-1]]
        swing_low = self.data.Low[lows[-1]]

        # Stop Loss (20 pips below/above 21 EMA)
        sl_long = ema_value - 0.002
        sl_short = ema_value + 0.002

        # Position Sizing
        position_size_long = calculate_position_size(self.equity, price, sl_long, self.risk_percent)
        position_size_short = calculate_position_size(self.equity, price, sl_short, self.risk_percent)

        # --- Trend Filtering: Only Trade When ADX > Threshold ---
        if adx_value < self.adx_threshold:
            return  # skips trading in low-trend environments

        # --- Long Entry ---
        if not self.position and price > swing_high and ema_value > self.ema21[-2]:  # EMA Sloping Up
            print(f"BUY Signal | Price: {price:.4f}, Size: {position_size_long}")
            self.buy(size=position_size_long, sl=sl_long)
            print(f"stop loss: {sl_long:.4f}") 
            print(f"Equity = {self.equity:.2f}") 
            print("---------------------------")

        # --- Short Entry ---
        elif not self.position and price < swing_low and ema_value < self.ema21[-2]:  # EMA Sloping Down
            print(f"SELL Signal | Price: {price:.4f}, Size: {position_size_short}")
            self.sell(size=position_size_short, sl=sl_short)
            print(f"stop loss: {sl_short:.4f}") 
            print(f"Equity = {self.equity:.2f}")
            print("---------------------------") 

        # --- Exit Condition: Trailing Stop at 21 EMA ---
        if self.position:
            price = self.data.Close[-1]  # Latest close price
            ema_value = self.ema21[-1]   # Current EMA value

            # Exit long position if price drops below EMA
            if self.position.is_long and price < ema_value:
                self.position.close()


            # Exit short position if price moves above EMA
            elif self.position.is_short and price > ema_value:
                self.position.close()

            





# --- fetch 5m Data for EURUSD ---

EURUSD = yf.download("EURUSD=X", interval="5m", start="2025-03-01", end="2025-03-09")

if isinstance(EURUSD.columns, pd.MultiIndex):
    EURUSD.columns = EURUSD.columns.droplevel(1)  

if EURUSD.columns[0] == "Close":
    EURUSD = EURUSD[['Open', 'High', 'Low', 'Close', 'Volume']]

EURUSD.tail()


backtest = Backtest(EURUSD, EURUSD_TradingStrategy, commission=0.002, margin=1/20)
stats = backtest.run()

print(stats)
trades = stats._trades
for trade in trades.itertuples():
    print(f"Entry: {trade.EntryPrice:.4f}, Exit: {trade.ExitPrice:.4f}, P&L: {trade.PnL:.2f}%")
backtest.plot()

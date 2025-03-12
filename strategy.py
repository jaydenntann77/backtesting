import pandas as pd

def SMA(array, n):
    return pd.Series(array).rolling(n).mean()

def RSI(array, n):
    gain = pd.Series(array).diff()
    loss = gain.copy()
    gain[gain < 0] = 0
    loss[loss > 0] = 0
    rs = gain.ewm(n).mean() / loss.abs().ewm(n).mean() #ratio of average gain to average loss over n periods
    return 100 - 100 / (1 + rs)

from backtesting import Strategy, Backtest
from backtesting.lib import resample_apply


class System(Strategy):
    d_rsi = 30  # Daily RSI lookback periods
    w_rsi = 30  # Weekly
    level = 70
    
    def init(self):
        # Compute moving averages
        self.ma10 = self.I(SMA, self.data.Close, 10)
        self.ma20 = self.I(SMA, self.data.Close, 20)
        self.ma50 = self.I(SMA, self.data.Close, 50)
        self.ma100 = self.I(SMA, self.data.Close, 100)
        
        # Compute daily RSI(30)
        self.daily_rsi = self.I(RSI, self.data.Close, self.d_rsi)
        
        # To construct weekly RSI, use `resample_apply()`, a helper function from the backtesting library
        self.weekly_rsi = resample_apply(
            'W-FRI', RSI, self.data.Close, self.w_rsi) #uses friday's close data
        
        
    def next(self):
        price = self.data.Close[-1]
        
        # If we don't already have a position, and
        # if all conditions are satisfied, enter long.
        if (not self.position and
            self.daily_rsi[-1] > self.level and
            self.weekly_rsi[-1] > self.level and
            self.weekly_rsi[-1] > self.daily_rsi[-1] and
            self.ma10[-1] > self.ma20[-1] > self.ma50[-1] > self.ma100[-1] and
            price > self.ma10[-1]):
            
            # Buy at market price on next open, but do
            # set 8% fixed stop loss.
            self.buy(sl=.92 * price)
        
        # If the price closes 2% or more below 10-day MA
        # close the position, if any.
        elif price < .98 * self.ma10[-1]:
            self.position.close()
        

import yfinance as yf
import pandas as pd

from backtesting.test import GOOG

GOOG.tail()

# Download Google stock data
GOOGLE = yf.download("GOOG", start="2018-01-01", end="2020-06-30", auto_adjust=True)

# Drop MultiIndex if necessary
if isinstance(GOOGLE.columns, pd.MultiIndex):
    GOOGLE.columns = GOOGLE.columns.droplevel(1)  # Drop 'GOOG' from column names

# Reorder columns if 'Close' is first
if GOOGLE.columns[0] == "Close":
    GOOGLE = GOOGLE[['Open', 'High', 'Low', 'Close', 'Volume']]  # Ensure correct order

# Now it should work with Backtest()
from backtesting import Backtest
print("running backtest")
backtest = Backtest(GOOGLE, System, commission=0.002)
stats = backtest.run()
print(stats)

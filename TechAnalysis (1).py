#!/usr/bin/env python
# coding: utf-8

# In[75]:


# import necessary libraries 

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import warnings
import yfinance as yf
warnings.filterwarnings('ignore')


# In[76]:


# import the closing price data of the Ultratech Cement stock for the period of 5 years -
# from 24th march 2018 to 24th march 2023

import pandas_datareader.data as web

start = datetime.datetime(2018, 3, 24)
end = datetime.datetime(2023, 3, 24)

ultratech_df = yf.download("ULTRACEMCO.NS", start="2018-03-24", end="2023-03-24")


# In[77]:


ultratech_df


# In[78]:


ultratech_df.isnull().sum()


# In[79]:


ultratech_df.shape


# In[80]:


# observing general price variation of the closing price for the give period
ultratech_df['Adj Close'].plot(figsize = (15, 8), fontsize = 12)
plt.grid()
plt.ylabel('Price in Rupees')
plt.show()


# In[81]:


#Visualize the Stock Price
plt.figure(figsize = (17, 6))
plt.plot(ultratech_df['Adj Close'], label = 'Close')
plt.xticks(rotation = 45)
plt.title("Close Price History")
plt.xlabel('Date')
plt.ylabel("Price ")
plt.show()


# In[82]:


#Calculate the MACD and signal line
#Calculate the short term exponential moving average (EMA)
ShortEMA = ultratech_df['Adj Close'].ewm(span=12, adjust = False).mean()
#Calculate the long term exponential moving average (EMA)
LongEMA = ultratech_df['Adj Close'].ewm(span=26, adjust=False).mean()
#Calculate the MACD Line
MACD = ShortEMA - LongEMA
#Calculate the signal line
signal = MACD.ewm(span = 9, adjust=False).mean()


# In[83]:


# Create a new figure
plt.figure(figsize=(12, 6))

# Plot the Adjusted Close Prices
#plt.plot(ultratech_df.index, ultratech_df['Adj Close'], label='Adj Close', color='blue')

# Plot the Short-term EMA
plt.plot(ShortEMA.index, ShortEMA, label='Short EMA (12)', color='green')

# Plot the Long-term EMA
plt.plot(LongEMA.index, LongEMA, label='Long EMA (26)', color='red')

# You can also plot the MACD and signal line for reference
# plt.plot(MACD.index, MACD, label='MACD', color='purple')
# plt.plot(signal.index, signal, label='Signal', color='orange')

plt.legend()
plt.title('Short-term and Long-term Exponential Moving Averages (EMA)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.show()


# In[84]:


#plot the figure
plt.figure(figsize=(17,6))
plt.plot(ultratech_df.index, MACD, label = 'MACD', color = 'red')
plt.plot(ultratech_df.index, signal, label='Signal Line', color = 'blue')
plt.legend(loc = 'upper left')
plt.show()


# In[85]:




# Assuming you have already calculated ShortEMA, LongEMA, MACD, and signal as mentioned in your code

# Calculate the MACD histogram
MACD_histogram = MACD - signal

# Create a DataFrame to hold the MACD and its histogram
macd_data = pd.DataFrame({'MACD': MACD, 'MACD Signal': signal, 'MACD Histogram': MACD_histogram})

# Plot the MACD Histogram
plt.figure(figsize=(12, 6))
plt.bar(macd_data.index, macd_data['MACD Histogram'], color='g', label='MACD Histogram')
plt.axhline(0, color='r', linestyle='--', label='Zero Line')
plt.legend()
plt.title('MACD Histogram')
plt.xlabel('Date')
plt.ylabel('MACD Histogram Value')
plt.show()


# In[86]:


ultratech_df['MACD'] = MACD
ultratech_df['Signal Line'] = signal

ultratech_df


# In[87]:


#Create a signal to when to buy and sell an assert
def buy_sell(signal):
  Buy = []
  Sell = []
  flag = -1

  for i in range(0, len(signal)):
    if signal['MACD'][i] > signal['Signal Line'][i]:
      Sell.append(np.nan)
      if flag != 1:
        Buy.append(signal['Close'][i])
        flag = 1
      else:
        Buy.append(np.nan)
    elif signal['MACD'][i] < signal['Signal Line'][i]:
      Buy.append(np.nan)
      if flag != 0:
        Sell.append(signal['Close'][i])
        flag = 0
      else:
        Sell.append(np.nan)
    else:
      Buy.append(np.nan)
      Sell.append(np.nan)
  return (Buy, Sell)


# In[88]:


#Create buy and sell column
a = buy_sell(ultratech_df)
ultratech_df['Buy_Signal_Price'] = a[0]
ultratech_df['Sell_Signal_Price'] = a[1]
     


# In[89]:


ultratech_df


# In[90]:


#Visualize the buy and sell signal

plt.figure(figsize=(14,6))
plt.scatter(ultratech_df.index, ultratech_df['Buy_Signal_Price'], color = 'green', label='Buy', marker='^', alpha=1)
plt.scatter(ultratech_df.index, ultratech_df['Sell_Signal_Price'], color = 'red', label='Sell', marker='v', alpha=1)
plt.plot(ultratech_df['Close'], label='Close Price', alpha=0.35)
plt.title('Close price Buy & Sell Signal')
plt.xlabel("Date")
plt.ylabel("Close price USD ($)")
plt.legend(loc='upper left')
plt.show()


# In[91]:


# Calculate daily returns based on the signals
ultratech_df['Daily_Return'] = ultratech_df['Adj Close'].pct_change() * ultratech_df['Signal Line'].shift(1)
ultratech_df['Daily_Return'] = ultratech_df['Adj Close'].pct_change() * ultratech_df['Signal Line'].shift(1)

# Calculate the cumulative returns
ultratech_df['Cumulative_Return'] = (1 + ultratech_df['Daily_Return']).cumprod()

ultratech_df['trade_return']=ultratech_df['Daily_Return']*ultratech_df['Adj Close']*100


# In[92]:


print(ultratech_df[['Adj Close', 'Signal Line', 'Daily_Return', 'Cumulative_Return','trade_return']])


# In[93]:


# Calculating benchmark return
initial_index = ultratech_df['Adj Close'].iloc[0]
final_index = ultratech_df['Adj Close'].iloc[-1]
benchmark_return = ((final_index - initial_index)/initial_index)*100
benchmark_return


# In[94]:


# calculating portfolio value
initial_investment = 100000
initial_price = ultratech_df['Adj Close'].iloc[0]
no_of_shares_held = initial_investment / initial_price
ultratech_df['portfolio_value'] = no_of_shares_held * ultratech_df['Adj Close']
no_of_shares_held
ultratech_df['portfolio_value']


# In[95]:


plt.figure(figsize=(10, 6))
plt.plot(ultratech_df.index, ultratech_df['portfolio_value'], label='portfolio_value')
plt.title('Portfolio Performance Over Time')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.show()


# In[96]:


# Get the stock returns of df
ultratech_df['stock_returns'] = ultratech_df['Close'].pct_change()
ultratech_df.dropna(subset=['stock_returns'], inplace=True)
ultratech_df


# In[97]:



# Plot stock returns in graph 
plt.figure(figsize=(10, 6))
plt.plot(ultratech_df.index, ultratech_df['stock_returns'], label='Stock Returns', color='blue')
plt.title('Stock Returns Over Time')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.legend()
plt.grid()
plt.show()

# Plot daily returns in graph
plt.figure(figsize=(10, 6))
plt.plot(ultratech_df.index, ultratech_df['Daily_Return'], label='Daily Returns', color='green')
plt.title('Daily Returns Over Time')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.legend()
plt.grid()
plt.show()


# In[98]:


# profit loss count and value
profit_count = sum(1 for result in ultratech_df['trade_return'] if result > 0)
loss_count = sum(1 for result in ultratech_df['trade_return'] if result < 0)
largest_profit_trade = ultratech_df['Daily_Return'].max()
largest_loss_trade = ultratech_df['Daily_Return'].min()

total_profit = sum(result for result in ultratech_df['trade_return'] if result > 0)
total_loss = sum(result for result in ultratech_df['trade_return'] if result < 0)
# Calculate the average of daily returns
avg_daily_return = ultratech_df['Daily_Return'].mean()

# Calculate the standard deviation of portfolio values
portfolio_volatility = ultratech_df['portfolio_value'].std()

print("mean",avg_daily_return)
print("std",portfolio_volatility)
print("Win ratio:", profit_count / len(ultratech_df['trade_return']))  
print("Total number of trades:", len(ultratech_df['trade_return']))
print("Number of profitable trades:", profit_count)
print("Number of losing trades:", loss_count)
print(f"Largest profit making trade: {largest_profit_trade}")
print(f"Largest Loss-making Trade:  {largest_loss_trade}")
print("Total profit:", total_profit)
print("Total loss:", total_loss)
print("end portfolio,:",ultratech_df['portfolio_value'])


# In[99]:


#strategy return
ending_value=ultratech_df['portfolio_value'][-1]
beginning_value=ultratech_df['portfolio_value'][0]
strategy_return = (ending_value - beginning_value) / beginning_value
strategy_return=strategy_return*100


# In[100]:


strategy_return


# In[101]:


#benchmark return
ending_value=ultratech_df['Adj Close'].iloc[-1]
beginning_value=ultratech_df['Adj Close'].iloc[0]
benchmark_return = (ending_value - beginning_value) / beginning_value
benchmark_return=benchmark_return*100
# Calculate the annualized returns
trading_days= 252
annualized_return = (1 + strategy_return) ** (trading_days / len(ultratech_df.index)) - 1
print(f"Annualized Return: {annualized_return:.2%}")


# In[102]:


print("benchmark returns:",benchmark_return)


# In[103]:


window = 252

# Calculate the max drawdown in the past window days for each day in the series.
# Use min_periods=1 if you want to let the first 252 days data have an expanding window
Roll_Max = ultratech_df['Adj Close'].rolling(window, min_periods=1).max()
Daily_Drawdown = ultratech_df['Adj Close']/Roll_Max - 1.0

# Next we calculate the minimum (negative) daily drawdown in that window.
# Again, use min_periods=1 if you want to allow the expanding window
Max_Daily_Drawdown = Daily_Drawdown.rolling(window, min_periods=1).min()

# Plot the results
Max_Daily_Drawdown
Daily_Drawdown.plot()
Max_Daily_Drawdown.plot()
plt.show()


# In[104]:


Roll_Max = ultratech_df['Adj Close'].cummax()
Daily_Drawdown = ultratech_df['Adj Close']/Roll_Max - 1.0
Max_Daily_Drawdown = Daily_Drawdown.cummin()


# In[105]:


print(Max_Daily_Drawdown)


# In[106]:


minimum_max_daily_drawdown = Max_Daily_Drawdown.min()


# In[107]:


print(minimum_max_daily_drawdown)


# In[108]:


minimum_max_daily_drawdown_percentage = minimum_max_daily_drawdown * 100

print("Minimum Max Daily Drawdown in Percentage:", minimum_max_daily_drawdown_percentage, "%")


# In[109]:



# Assuming you have returns data in the 'Daily_Return' column of 'ultratech_df'
returns = ultratech_df['Daily_Return']

# Calculate mean log returns
mean_log_returns = np.log(returns + 1).mean()

# Calculate mean returns
mean_returns = np.exp(mean_log_returns) - 1

# Calculate the standard deviation of returns
std = returns.std()

# Calculate the Sharpe ratio
sharpe_ratio = mean_returns / std * np.sqrt(252)  # Assuming 252 trading days in a year


# In[110]:


print(sharpe_ratio)


# In[111]:


ultratech_df.to_csv('Technical.csv')


# In[ ]:





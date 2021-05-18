#!/usr/bin/env python
# coding: utf-8

# # Unit 5 - Financial Planning
# 

# In[23]:


# Initial imports
import os
import requests
import pandas as pd
from dotenv import load_dotenv
import numpy as np
import alpaca_trade_api as tradeapi
from MCForecastTools import MCSimulation
import io

get_ipython().run_line_magic('matplotlib', 'inline')


# In[24]:


# Load .env enviroment variables
load_dotenv()


# ## Part 1 - Personal Finance Planner

# ### Collect Crypto Prices Using the `requests` Library

# In[25]:


# Set current amount of crypto assets
my_btc = 1.2
my_eth = 5.3


# In[26]:


# Crypto API URLs
btc_url = "https://api.alternative.me/v2/ticker/Bitcoin/?convert=USD"
eth_url = "https://api.alternative.me/v2/ticker/Ethereum/?convert=USD"
response_btc_url = requests.get(btc_url)
response_eth_url = requests.get(eth_url)


# In[27]:


response_btc = response_btc_url.json()
response_eth = response_eth_url.json()


# In[28]:


# Fetch current BTC price
current_btc_price = response_btc['data']['1']['quotes']['USD']['price']

# Fetch current ETH price
current_eth_price = response_eth['data']['1027']['quotes']['USD']['price']

# Compute current value of my crpto
my_btc_value = current_btc_price * my_btc
my_eth_value = current_eth_price * my_eth
total_crypto = my_btc_value + my_eth_value

# Print current crypto wallet balance
print(f"The current value of your {my_btc} BTC is ${my_btc_value:0.2f}")
print(f"The current value of your {my_eth} ETH is ${my_eth_value:0.2f}")
print(f"The total value of your crypto is ${total_crypto:0.2f}")


# ### Collect Investments Data Using Alpaca: `SPY` (stocks) and `AGG` (bonds)

# In[29]:


# Current amount of shares
my_agg = 200
my_spy = 50


# In[30]:


# Set Alpaca API key and secret
alpaca_api_key = os.getenv("ALPACA_API_KEY")
alpaca_secret_key = os.getenv("ALPACA_API_SECRET")
# Create the Alpaca API object
alpaca_trade = tradeapi.REST(alpaca_api_key,alpaca_secret_key)
type(alpaca_api_key)


# In[31]:


# Format current date as ISO format
start = pd.Timestamp("2021-05-14", tz= 'US/Pacific').isoformat()
end = pd.Timestamp("2021-05-14", tz= 'US/Pacific').isoformat()

# Set the tickers
tickers = ["AGG", "SPY"]

# Set timeframe to '1D' for Alpaca API
timeframe = "1D"

# Get current closing prices for SPY and AGG
agg_df = alpaca_trade.get_barset(tickers, timeframe, start=start, end=end).df
spy_df = alpaca_trade.get_barset(tickers, timeframe, start=start, end=end).df
# Preview DataFrame
agg_df
spy_df


# In[32]:


# Pick AGG and SPY close prices
agg_close_price =agg_df["AGG"]["close"][-1]
spy_close_price =spy_df["SPY"]["close"][-1]

# Print AGG and SPY close prices
print(f"Current AGG closing price: ${agg_close_price}")
print(f"Current SPY closing price: ${spy_close_price}")


# In[33]:


# Compute the current value of shares
my_agg_value = agg_close_price * my_agg
my_spy_value = spy_close_price * my_spy
total_stocks = my_agg_value + my_spy_value


# Print current value of share
print(f"The current value of your {my_spy} SPY shares is ${my_spy_value:0.2f}")
print(f"The current value of your {my_agg} AGG shares is ${my_agg_value:0.2f}")
print(f"The current value of your stock shares are ${total_stocks:0.2f}")


# ### Savings Health Analysis

# In[34]:


# Set monthly household income
monthly_income = 12000

# Create savings DataFrame
df_savings = pd.DataFrame(data=[total_crypto, total_stocks],index=['crypto', 'shares'], columns=['amount'])

# Display savings DataFrame
display(df_savings)


# In[35]:


# Plot savings pie chart
df_savings.plot(subplots=True,kind="pie",title="Crypto shares vs Stock Shares")


# In[36]:


# Set ideal emergency fund
emergency_fund = monthly_income * 3

# Calculate total amount of savings
total_savings = total_stocks + total_crypto
cover_amount = emergency_fund - total_savings
# Validate saving health
if total_savings >= emergency_fund:
    print("Congratulations! You have enough money in your emergency fund!")
else:
    total_savings < emergency_fund
    print(f"You are currently ${cover_amount} away from reaching your goal. ")


# ## Part 2 - Retirement Planning
# 
# ### Monte Carlo Simulation

# In[37]:


# Set start and end dates of five years back from today.
# Sample results may vary from the solution based on the time frame chosen
start_date = pd.Timestamp('2015-08-07', tz='America/New_York').isoformat()
end_date = pd.Timestamp('2020-08-07', tz='America/New_York').isoformat()


# In[38]:


# Get 5 years' worth of historical data for SPY and AGG
tickers = ["SPY","AGG"]

df_stock_data =alpaca_trade.get_barset(tickers, timeframe, start= start_date,end= end_date, limit=1000 ).df

# Display sample data
df_stock_data.head()


# In[39]:


# Configuring a Monte Carlo simulation to forecast 30 years cumulative returns
mc_thirtyyear = MCSimulation(df_stock_data,[0.6,0.4], 500, 252*30)


# In[40]:


# Printing the simulation input data
mc_thirtyyear.portfolio_data.head()


# In[41]:


# Running a Monte Carlo simulation to forecast 30 years cumulative returns
mc_thirtyyear.calc_cumulative_return()


# In[42]:


# Plot simulation outcomes
mc_thirtyyear.plot_simulation()


# In[43]:


# Plot probability distribution and confidence intervals
mc_thirtyyear.plot_distribution()


# ### Retirement Analysis

# In[44]:


# Fetch summary statistics from the Monte Carlo simulation results
mcsummary_statistics = mc_thirtyyear.summarize_cumulative_return()
# Print summary statistics
print(mcsummary_statistics)


# ### Calculate the expected portfolio return at the 95% lower and upper confidence intervals based on a `$20,000` initial investment.

# In[46]:


# Set initial investment
initial_investment = 20000

# Use the lower and upper `95%` confidence intervals to calculate the range of the possible outcomes of our $20,000
ci_lower = mc_thirtyyear.summarize_cumulative_return()[8] * initial_investment
ci_upper = mc_thirtyyear.summarize_cumulative_return()[9] * initial_investment

# Print results
print(f"There is a 95% chance that an initial investment of ${initial_investment} in the portfolio"
      f" over the next 30 years will end within in the range of"
      f" ${ci_lower:0.00f} and ${ci_upper:0.00f}")


# ### Calculate the expected portfolio return at the `95%` lower and upper confidence intervals based on a `50%` increase in the initial investment.

# In[47]:


# Set initial investment
initial_investment = 30000 * 1.5

# Use the lower and upper `95%` confidence intervals to calculate the range of the possible outcomes of our $30,000
ci_lower = (mc_thirtyyear.summarize_cumulative_return()[8] * initial_investment) * 1.5
ci_upper = (mc_thirtyyear.summarize_cumulative_return()[9] * initial_investment) * 1.5

# Print results
print(f"There is a 95% chance that an initial investment of ${initial_investment} in the portfolio"
      f" over the next 30 years will end within in the range of"
      f" ${ci_lower:0.0f} and ${ci_upper:0.0f}")


# ## Optional Challenge - Early Retirement
# 
# 
# ### Five Years Retirement Option

# In[48]:


# Configuring a Monte Carlo simulation to forecast 5 years cumulative returns
mc_fiveyear = MCSimulation(df_stock_data,[0.85,0.15], 500, 252*5)


# In[49]:


# Running a Monte Carlo simulation to forecast 5 years cumulative returns
print(mc_fiveyear)


# In[50]:


# Plot simulation outcomes
mc_fiveyear.plot_simulation()


# In[51]:


# Plot probability distribution and confidence intervals
mc_fiveyear.plot_distribution()


# In[52]:


# Fetch summary statistics from the Monte Carlo simulation results
mcsummary_statistics_fiveyear = mc_fiveyear.summarize_cumulative_return()

# Print summary statistics
mcsummary_statistics_fiveyear


# In[53]:


# Set initial investment
initial_investment2 = 60000
# Use the lower and upper `95%` confidence intervals to calculate the range of the possible outcomes of our $60,000
ci_lower_five = mc_fiveyear.summarize_cumulative_return()[8] * initial_investment2
ci_upper_five = mc_fiveyear.summarize_cumulative_return()[9] * initial_investment2

# Print results
print(f"There is a 95% chance that an initial investment of ${initial_investment} in the portfolio"
      f" over the next 5 years will end within in the range of"
      f" ${ci_lower_five:0.0f} and ${ci_upper_five:0.0f}")


# ### Ten Years Retirement Option

# In[54]:


# Configuring a Monte Carlo simulation to forecast 10 years cumulative returns
mc_tenyear = MCSimulation(df_stock_data,[0.6,0.4], 500, 252*10)


# In[55]:


# Running a Monte Carlo simulation to forecast 10 years cumulative returns
print(mc_tenyear)


# In[56]:


# Plot simulation outcomes
mc_tenyear.plot_simulation()


# In[57]:


# Plot probability distribution and confidence intervals
mc_tenyear.plot_distribution()


# In[58]:


# Fetch summary statistics from the Monte Carlo simulation results
mc_tenyear_return = mc_tenyear.summarize_cumulative_return()
# Print summary statistics
print(mc_tenyear_return)


# In[59]:


# Set initial investment
initial_investment = 60000
# Use the lower and upper `95%` confidence intervals to calculate the range of the possible outcomes of our $60,000
ci_lower_ten = mc_tenyear.summarize_cumulative_return()[8] * initial_investment
ci_upper_ten = mc_tenyear.summarize_cumulative_return()[9] * initial_investment
# Print results
print(f"There is a 95% chance that an initial investment of ${initial_investment} in the portfolio"
      f" over the next 10 years will end within in the range of"
      f" ${ci_lower_ten:0.0f} and ${ci_upper_ten:0.0f}")


# In[ ]:





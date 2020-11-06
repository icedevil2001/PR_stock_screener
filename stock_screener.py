#stock_screener.py


from yahoo_fin import stock_info as si
import yfinance as yf
import pandas as pd
import requests
import datetime
import time
from collections import OrderedDict
import streamlit as st
import base64

def period(days=365):
  '''
  return start and end dates
  '''
  start_date = datetime.datetime.now() - datetime.timedelta(days=365)
  end_date = datetime.date.today()
  return start_date, end_date 

def calc_relative_strength(df):

  # if "adj_close" in df.columns:

  if isinstance(df, pd.Series):
    ## relative gain and losses
    df = df.rename('adj_close').to_frame()
  df['close_shift'] = df['adj_close'].shift(1)
  ## Gains (true) and Losses (False)
  df['gains'] = df.apply(lambda x: x['adj_close'] if x['adj_close'] >= x['close_shift'] else 0, axis=1)
  df['loss'] = df.apply(lambda x: x['adj_close'] if x['adj_close'] <= x['close_shift'] else 0, axis=1)

  avg_gain = df['gains'].mean()
  avg_losses = df['loss'].mean()

  return avg_gain / avg_losses

def ftsc100():
	fstc = pd.read_html('https://en.wikipedia.org/wiki/FTSE_100_Index')[3].loc[:, 'EPIC'].to_list()
	return [f'{x.replace(".", "")}.L' for x in fstc]
	
def rs_rating(stock_rs_strange_value, index_rs_strange_value):
  # print(f'Stock RS:{stock_rs_strange_value}, Index RS:{index_rs_strange_value}')
  return 100 * ( stock_rs_strange_value / index_rs_strange_value )

def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="MM_stock_screener.csv">Download CSV File</a>'
    return href    

def get_stock(stocklist, days=365):
	start_date, end_date =period(days)
	df = yf.download(stocklist, start=start_date, end=end_date)
	df = df.drop(['High', 'Low', 'Open','Close'], axis=1)
	df = df.rename(columns={'Adj Close': "adj_close"})
	return df

def get_index_companies(index_tinker ):
  index_stocklist = {
    'DOW': si.tickers_sp500(),
    'NASDAQ': si.tickers_nasdaq(),
    "S&P500": si.tickers_sp500(),
    'FTSE100': ftsc100()
    }
  if isinstance(index_tinker, str):
    stocklist = index_stocklist.get(index_tinker)
    if stocklist:
      return stocklist
    else:
      return [index_tinker]
  else: 
    return index_tinker

def get_company_name(tinker):
  '''
  Get the company name if possible
  '''
  try:
    return yf.Ticker(tinker).info['longName']
  except:
    return tinker

def index_tinker(index_symbol):
  
  index_tinker = {
    'DOW': 'DOW',
    'NASDAQ': '^IXIC', 
    "S&P500": '^GSPC',
    'FTSE100': '^FTSE'
    }
  return index_tinker.get(index_symbol)

def SMA_50(df):
  '''
  Simple moving average (SMA) over 50days
  '''
  return round(df.rolling(window=50).mean(), 2)[-1]

def SMA_150(df):
  '''
  Simple moving average (SMA) over 150days
  '''
  return round(df.rolling(window=150).mean(), 2)[-1]

def SMA_200(df):
  '''
  Simple moving average (SMA) over 200days
  '''
  return round(df.rolling(window=200).mean(), 2)[-1]

def Low_52week(df):
  '''
  Lowest price of 52 weeks
  '''
  return df.iloc[-260:].min()

def High_52week(df):
  '''
  High price of 52 weeks
  '''
  return df.iloc[-260:].min()

def Current_Price(df):
  return df.iloc[-1] 

def Avg_Price_Month(df):
  return df.iloc[-20:].mean()

def SMA_200_20(df):
  '''
  Simple moving average (SMA) over 200days at 1month (20days week days)
  '''
  val = round(df.rolling(window=200).mean() )
  return val[-20]

def Perc_Change_Over_Year(df):
  '''
  Calculate percentage change over the year
  '''
  return 100 * (df.iloc[-1] / df.iloc[0])

def Avg_Volume(df):
  ''' 
  Calculate average volumne df['volume']
  '''
  return df.mean()

## -- Conditions -- ##
def condition1(df):
  # Condition 1: Current Price > 150 SMA and > 200 SMA
  if (df.Current_Price > df.SMA_150 and df.Current_Price > df.SMA_200):
    return True
  return False

def condition2(df):
  # Condition 2: 150 SMA and > 200 SMA
  if (df.SMA_150 > df.SMA_200):
    return True
  return False

def condition3(df):
  # Condition 3: 200 SMA trending up for at least 1 month (ideally 4-5 months)
  if df.SMA_200 > df.SMA_200_20:
    return True 
  return False

def condition4(df):
  # Condition 4: 50 SMA> 150 SMA and 50 SMA> 200 SMA
  if df.SMA_50 > df.SMA_150 > df.SMA_200:
    return True
  return False

def condition5(df):
  # Condition 5: Current Price > 50 SMA
  if df.Current_Price > df.SMA_50:
    return True 
  return False

def condition6(df):
  # Condition 6: Current Price is at least 30% above 52 week low (Many of the best are up 100-300% before coming out of consolidation)
  if df.Current_Price >= (1.3 * df.Low_52week):
    return True
  return False

def condition7(df):
  # Condition 7: Current Price is within 25% of 52 week high
  if df.Current_Price >= (0.75 * df.High_52week):
    return True
  return False

def condition8(df, min_rs_rating=70):
  # Condiction 8: IBD RS_Rating greater than 70
  if df.RS_rating >= min_rs_rating:
    return True
  return False

def min_average_volume(df, min_vol=1000):
		if df.Average_Volume> min_vol:
		    return True
		return False
## -- End Condition -- ##


def stock_screener(index_symbol, min_volume=1, min_price=10, days=365, min_rs_rating=70):

	# index_symbol = 'FTSE100'
	# iteration = st.empty()

	# iteration.text(f'Stocks Processed: {(num+1)}/{total}')

	st.text('Please be patient, it may take several minutes..')

	## Get index data
	index_df = get_stock(index_tinker(index_symbol))

	## calculate index RS strength 
	index_rs_strength = calc_relative_strength(index_df.loc[:,'adj_close'])
	# index_rs_strength


	stocklist = get_index_companies(index_symbol)

	df = get_stock(stocklist)
	df = df.dropna(axis=0, how='all')

	st.text(f'Total stocks downloaded from {index_symbol} {df.shape[1]/2}')
	st.text(f'Processing stocks..')
	## Calculation on each column 
	final = (df['adj_close']
			 	.apply( [
			          Current_Price,
			          Avg_Price_Month,
			          Perc_Change_Over_Year,
			          lambda x: rs_rating(calc_relative_strength(x), index_rs_strength),
			          SMA_50,
			          SMA_150,
			          SMA_200,
			          SMA_200_20,
			          Low_52week,
			          High_52week
			    ], 
			    axis=0
			  )
			)


	## rename lambda func
	final = final.T.rename(columns={'<lambda>': "RS_rating"} ).T

	## Calcualte average volyme
	avg_vol = df['Volume'].apply(Avg_Volume)
	avg_vol = avg_vol.rename('Average_Volume').to_frame().T
	
	## add the average volum to df 
	final = final.append(avg_vol)

	## Check which columns (stocks) meet conditions
	meet_conditions = final.apply(
	    [
	     condition1,
	     condition2,
	     condition3,
	     condition4,
	     condition5,
	     condition6,
	     condition7,
	     condition8,
	     min_average_volume,
	     
	     ])
	# meet_conditions

	## select stocks that have meet all conditions
	pass_conditions_idx = [k for k,v in meet_conditions.all().items() if v]

	final = final.T
	final = final.loc[final.index.isin(pass_conditions_idx)]
	final.index = final.index.rename('ticker')
	final = final.reset_index()
	st.text(f'{len(final)} stocked passed Mark Minervin\'s conditions')
	final['Company_Name'] = final.ticker.map(get_company_name)

	column_order  = ['ticker','Company_Name', 'Current_Price','Average_Volume','Avg_Price_Month', 'Perc_Change_Over_Year',
	       'RS_rating', 'SMA_50', 'SMA_150', 'SMA_200', 'SMA_200_20', 'Low_52week',
	       'High_52week']
	final = final[column_order]
	final = final.round(2) ## round floats 

	return final #.query('(Current_Price >= @min_price) & (Average_Volume >= @min_volume) & (RS_rating >= @min_rs_rating)')


########

#### ---- The App ---- ####
## ref: https://towardsdatascience.com/making-a-stock-screener-with-python-4f591b198261
st.sidebar.header('Settings')
index_symbol = st.sidebar.selectbox('Index', ['FTSE100', 'S&P500', 'DOW', 'NASDAQ' ] )
min_volume = st.sidebar.text_input("Minimum Volume", 1e6)
min_price = st.sidebar.slider('Minimum Price ($)', 0,5000, 0)
days = st.sidebar.slider('Max Period (days)', 14, 730, 365)
min_rs_rating = st.sidebar.slider('Minimum Relative Strange Rating', 1, 100, 70)

with st.beta_container():
	st.title('Mark Minervini’s Trend stock screener')
	st.write('''
		I've created this app help screen for stock using the Mark Minervini's 8 principles
		inspried by these blogs:
		* [How To Scan Mark Minervini’s Trend Template Using Python](https://www.marcellagerwerf.com/how-to-scan-mark-minervinis-trend-template-using-python/)
		* [How to build a stock screener](https://www.youtube.com/watch?v=hngHA9Jjbjc&list=PLPfme2mwsQ1FQhH1icKEfiYdLSUHE-Wo5&index=3&ab_channel=RichardMoglen)
		* [Making a Stock Screener with Python!](https://towardsdatascience.com/making-a-stock-screener-with-python-4f591b198261)

		You can read more about this template in Mark Minervini’s [blog post](http://www.minervini.com/blog/index.php/blog/first_things_first_how_to_chart_stocks_correctly_and_increase_your_chances).
		''')
	expander = st.beta_expander("Principles")

	expander.write('''
	
		1. The current price of the security must be greater than the 150 and 200-day simple moving averages.
		2. The 150-day simple moving average must be greater than the 200-day simple moving average.
		3. The 200-day simple moving average must be trending up for at least 1 month.
		4. The 50-day simple moving average must be greater than the 150 simple moving average and the 200 simple moving average.
		5. The current price must be greater than the 50-day simple moving average.
		6. The current price must be at least 30% above the 52 week low.
		7. The current price must be within 25% of the 52 week high.
		8. The IBD RS-Rating must be greater than 70 (the higher, the better). The RS rating is a metric of a stock’s price performance over the last year compared to all other stocks and the overall market. Check out this article to learn more.
	
	''')
	# I created this article to help others make an easy-to-read stock screener Python program based on Mark Minervini’s Trend Template (the 8 principles on selecting the best stocks


	if st.button('Start screening'):
		st.header(f'Screen Stock for {index_symbol}')

		final_df = stock_screener(index_symbol, min_volume, min_price, days, min_rs_rating)
		st.dataframe(final_df)
		
		st.markdown(filedownload(final_df), unsafe_allow_html=True)
		
		st.set_option('deprecation.showPyplotGlobalUse', False)








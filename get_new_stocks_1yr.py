import pandas as pd
import yfinance as yf
import pickle
import requests
import lxml
import bs4 as bs
from datetime import datetime

# Define the S&P 500 tickers
# For demonstration purposes, using a small subset of tickers. Replace with the full list of S&P 500 tickers.
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'FB']
resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')        
soup = bs.BeautifulSoup(resp.text,'lxml')        
table = soup.find('table', {'class': 'wikitable sortable'})        

tickers = []

for row in table.findAll('tr')[1:]:
    ticker = row.findAll('td')[0].text
    tickers.append(ticker)

with open("sp500tickers.pickle", "wb") as f:
    pickle.dump(tickers, f)
print(tickers)

# Define the time period
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - pd.DateOffset(years=1)).strftime('%Y-%m-%d')

end_date = '2024-07-05'
start_date = '2019-01-01'

# Initialize an empty list to collect data
all_data = []

# Pull data for each ticker
for ticker in tickers:
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data['Name'] = ticker
    all_data.append(stock_data)

# Concatenate all data into a single DataFrame
data2 = pd.concat(all_data).reset_index()

# Save the DataFrame to a CSV file
data2.to_csv('2019-2024-stock-data.csv', index=False)
print(data2.head())
print("Data successfully pulled and saved to 2013-2019-stock-data.csv")

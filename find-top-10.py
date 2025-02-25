import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from datetime import datetime
import joblib
import json
import math
import matplotlib.pyplot as plt

# Step 1: Load the Serialized Models
with open('serialized_models.json', 'r') as fin:
    serialized_models = json.load(fin)
models = {name: model_from_json(model_json) for name, model_json in serialized_models.items()}
print("Models loaded from serialized_models.json")

# Step 2: Get Recent Data - from get_new_stocks_1yr.py
data2 = pd.read_csv('2019-2024-regressors.csv')
data2['Name'] = data2['Name'].str.strip()

# Debugging: Check the first few rows of data2
print("Recent data preview:")
print(data2.head())

# Calculate the 30-day moving average for each stock in recent data
#data2['moving_avg'] = data2.groupby('Name')['y'].transform(lambda x: x.rolling(window=30, min_periods=1).mean())
print(data2.tail(5))

# Step 3: Forecast Future Prices using each model's make_future_dataframe
predictions = {}

for name, model in models.items():
    # Make future dataframe
    future = model.make_future_dataframe(periods=5)
    
    # Ensure the stock data is not empty and has enough data points
    stock_data = data2[data2['Name'] == name]
    if not stock_data.empty and len(stock_data) > 0:
        # Add all regressors to the future dataframe
        for regressor in ['moving_avg', 'RSI', 'MACD', 'Signal_Line', 'Upper_BB', 'Lower_BB', 'volatility', 'log_return']:
            if not stock_data[regressor].empty and not pd.isna(stock_data[regressor].iloc[-1]):
                future[regressor] = stock_data[regressor].iloc[-1]
            else:
                print(f"Skipping {name} due to missing or NaN value in {regressor}.")
                break
        else:  # This else belongs to the for loop, it runs if the loop completes without break
            # Predict the future prices
            forecast = model.predict(future)
            forecast['Name'] = name
            predictions[name] = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            print(f"Parsed {name}.")
    else:
        print(f"Skipping {name} due to insufficient data.")
# Step 4: Identify Top Stocks
predicted_increases = []

for name, forecast in predictions.items():
    print(f"Processing {name}")
    
    stock_data = data2[data2['Name'] == name]
    
    if not stock_data.empty:
        recent_price = stock_data['y'].iloc[-1]
        future_price = forecast[forecast['ds'] == forecast['ds'].max()]['yhat'].values[0]
        increase_percentage = (future_price - recent_price) / recent_price * 100
        predicted_increases.append((name, increase_percentage))
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(1)
    else:
        print(f"No data found for stock: {name}. Skipping.")

# Step 5: Sort by the predicted increase percentage and select the top 10 stocks
top_stocks = sorted(predicted_increases, key=lambda x: x[1], reverse=True)[:15]

# Step 6: Print the top 10 stocks
#print("Top 10 stocks predicted to increase in value over the next month:")
#for stock in top_stocks:
   #print(f"Stock: {stock}, \nPredicted Increase: {stock[1]:.2f}%")
    #fig1 = predictions[name].plot(forecast[stock])
    #fig1
    
for name in top_stocks:
    print(f"Stock: {name}, \nPredicted Increase: {name[1]:.2f}%")
    model = models[name[0]]
    forecast = predictions[name[0]]
    
    fig = model.plot(forecast).savefig('graphs/'+name[0]+'.png')
    fig
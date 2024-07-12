import pandas as pd
import yfinance as yf
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
from datetime import datetime
import joblib
import json

# Step 1: Load the Serialized Models
with open('serialized_models.json', 'r') as fin:
    serialized_models = json.load(fin)
models = {name: model_from_json(model_json) for name, model_json in serialized_models.items()}
print("Models loaded from serialized_models.json")

# Step 2: Get Recent Data - from get_new_stocks_1yr.py
data2 = pd.read_csv('snp500_1yr.csv')
data2['Name'] = data2['Name'].str.strip()

# Debugging: Check the first few rows of data2
print("Recent data preview:")
print(data2.head())

# Step 3: Forecast Future Prices
# Using the future dataframe created from one of the loaded models
future = list(models.values())[0].make_future_dataframe(periods=10)
predictions = {}

for name, model in models.items():
    forecast = model.predict(model.make_future_dataframe(periods=10))
    forecast['Name'] = name
    predictions[name] = forecast[['ds', 'yhat']]

# Step 4: Identify Top Stocks
predicted_increases = []

for name, forecast in predictions.items():
    print(f"Processing {name}")
    
    stock_data = data2[data2['Name'] == name]
    
    if not stock_data.empty and not stock_data['y'].empty:
        recent_price = stock_data['y'].iloc[-1]
        future_price = forecast[forecast['ds'] == future['ds'].max()]['yhat'].values[0]
        increase_percentage = (future_price - recent_price) / recent_price * 100
        predicted_increases.append((name, increase_percentage))
        print(f"Stock: {name} - Increase %: {increase_percentage:.2f} - recent price: {recent_price} - expected price - {future_price}")
    else:
        print(f"No data found for stock: {name}. Skipping.")

# Step 5: Sort by the predicted increase percentage and select the top 10 stocks
top_stocks = sorted(predicted_increases, key=lambda x: x[1], reverse=True)[:25]

# Step 6: Print the top 10 stocks
print("Top 10 stocks predicted to increase in value over the next month:")
for stock in top_stocks:
    print(f"Stock: {stock}, \nPredicted Increase: {stock[1]:.2f}%")

import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
import joblib
import json
from itertools import product
from prophet.diagnostics import cross_validation, performance_metrics
import numpy as np

def add_features(df):
    # RSI
    delta = df['y'].diff()
    gain = (delta.where(delta >= 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['y'].ewm(span=12, adjust=False).mean()
    exp2 = df['y'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df['MA20'] = df['y'].rolling(window=20).mean()
    df['SD20'] = df['y'].rolling(window=20).std()
    df['Upper_BB'] = df['MA20'] + (df['SD20']*2)
    df['Lower_BB'] = df['MA20'] - (df['SD20']*2)
    
    # Add more features
    df['log_return'] = np.log(df['y']).diff()
    df['volatility'] = df['log_return'].rolling(window=30).std() * np.sqrt(252)
    
    # Add day of week and month features
    df['day_of_week'] = df['ds'].dt.dayofweek
    df['month'] = df['ds'].dt.month
    
    df = df.ffill()
    df = df.bfill()
    
    return df
     
def perform_cv(model, group):
    df_cv = cross_validation(model, initial='730 days', period='180 days', horizon='30 days')
    df_p = performance_metrics(df_cv)
    return df_p['mape'].mean()

def tune_prophet(group):
    param_grid = {
        'changepoint_prior_scale': [0.002, 0.05, 0.2, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
        'holidays_prior_scale': [0.02, 0.2, 2.0, 20.0],
        'seasonality_mode': ['additive', 'multiplicative']
    }

    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
    
    mapes = []  # Store the MAPE values for each combination of parameters
    
    for params in all_params:
        m = Prophet(**params)
        for regressor in ['moving_avg', 'RSI', 'MACD', 'Signal_Line', 'Upper_BB', 'Lower_BB', 'volatility', 'log_return']:
            m.add_regressor(regressor)
        m.fit(group)
        mape = perform_cv(m, group)
        mapes.append(mape)
    
    # Find the best parameters
    best_params = all_params[mapes.index(min(mapes))]
    
    return best_params
    

# Step 1: Prepare the Training Data

# Load and prepare the 5-year training data
data_5yr = pd.read_csv('2019-2024-stock-data.csv')
data_5yr['ds'] = pd.to_datetime(data_5yr['Date'])
data_5yr['y'] = data_5yr['Close']
data_5yr = data_5yr[['ds', 'y', 'Name']]

# Load the recent 1-year data
#data_1yr = pd.read_csv('snp500_1yr.csv')
#data_1yr['ds'] = pd.to_datetime(data_1yr['ds'])
#data_1yr['y'] = data_1yr['close']
#data_1yr = data_1yr[['ds', 'y', 'Name']]

# Combine the datasets
#data = pd.concat([data_5yr, data_1yr], ignore_index=True)
data = data_5yr
data['Name'] = data['Name'].str.strip()


# Debugging: Check the first few rows after formatting

data['moving_avg'] = data.groupby('Name')['y'].transform(lambda x: x.rolling(window=30, min_periods=1).mean())
data = data.groupby('Name').apply(add_features).reset_index(drop=True)
print("Data format:")
print(data.head())

# For demonstration, filling NaN values in RSI column
data['RSI'] = data.groupby('Name')['y'].transform(lambda x: (x - x.min()) / (x.max() - x.min()) * 100)

data.to_csv('2019-2024-regressors.csv', index=False)
print("Data saved to 2013-2019-regressors.csv")

# Step 2: Train the FB Prophet Model
models = {}
for name, group in data.groupby('Name'):
    if group.shape[0] > 730:  # Ensure enough data for CV and tuning
        best_params = tune_prophet(group[['ds', 'y', 'moving_avg', 'RSI', 'MACD', 'Signal_Line', 'Upper_BB', 'Lower_BB', 'volatility', 'log_return']])
        model = Prophet(**best_params)
        for regressor in ['moving_avg', 'RSI', 'MACD', 'Signal_Line', 'Upper_BB', 'Lower_BB', 'volatility', 'log_return']:
            model.add_regressor(regressor)
        model.fit(group[['ds', 'y', 'moving_avg', 'RSI', 'MACD', 'Signal_Line', 'Upper_BB', 'Lower_BB', 'volatility', 'log_return']])
        mape = perform_cv(model, group)
        print(f"{name} MAPE: {mape}")
        models[name] = model
    else:
        print(f"Skipping {name} due to insufficient data.")
        
# Step 3: Serialize the Models to JSON
serialized_models = {name: model_to_json(model) for name, model in models.items()}
with open('serialized_models.json', 'w') as fout:
    json.dump(serialized_models, fout)
print("Models serialized and saved to serialized_models.json")


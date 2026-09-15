"""
Student Name: Oliver Wuttke
Student FAN: WUTT0019
File: questions_1-5.py
Date: 15-09-2026
Description: Contains the code for questions 1-5 for Lab 6.
"""

import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

# === Question 1 ===
# Setting up for our API request
endpoint = 'https://archive-api.open-meteo.com/v1/archive'
params = {
    'latitude': -34.9287,   # Latitude for Adelaide
    'longitude': 138.5986,  # Longitude for Adelaide
    'start_date': '2026-06-01',
    'end_date': '2026-08-31',
    'hourly': 'temperature_2m',
    'timezone': 'Australia/Adelaide'
}

# Making request
response = requests.get(endpoint, params=params)
wdata = response.json()

# Extracting relevant data
timestamps = [datetime.fromisoformat(item) for item in wdata['hourly']['time']]
temperatures = wdata['hourly']['temperature_2m']

# Creating a pandas dataframe
wdf = pd.DataFrame({
    'Date': timestamps,
    'Temp': temperatures
})

# Setting date column as index
wdf.set_index('Date', inplace=True)

# Set frequency to hourly
wdf = wdf.asfreq('h')

# Seasonal decomposition
decomp = seasonal_decompose(wdf['Temp'], model='additive', period=24)

# Plotting decomposed components
decomp.plot()
plt.show()

# === Question 2 ===
# ARIMA model init and fitting
arima_model = ARIMA(wdf['Temp'], order=(5, 1, 0))
arima_model_fit = arima_model.fit()

# Compute Root Mean Squared Error
train_rmse = np.sqrt((arima_model_fit.resid[1:] ** 2).mean())
print(f'RMSE: {train_rmse}')

# === Question 3 ===
# ARIMA model init and fitting for different orders
orders = [
    (5, 1, 0),
    (1, 1, 1),
    (2, 1, 2)
]

for order in orders:
    arima_model = ARIMA(wdf['Temp'], order=order)
    arima_model_fit = arima_model.fit()

    train_rmse = np.sqrt((arima_model_fit.resid[1:] ** 2).mean())
    print(f'RMSE for order {order}: {train_rmse}')

# === Question 4 ===
# Train test split
train, test = wdf['Temp'][:-7], wdf['Temp'][-7:]

# Fit model
arima_model = ARIMA(train, order=(5, 1, 0)).fit()

# Rolling forecast
predictions = []
for t in test:
    predictions.append(arima_model.forecast())
    arima_model = arima_model.append([t], refit=False)

# Compute metrics
print('Mean Absolute Error: ', mean_absolute_error(test, predictions))
print('Mean Squared Error: ', mean_squared_error(test, predictions))
print('Root Mean Squared Error: ', root_mean_squared_error(test, predictions))
print('R^2 Score: ', r2_score(test, predictions))

# === Question 5 ===
# Hour to integer exogenous variable
exog = wdf.index.hour.to_numpy().reshape(-1, 1)

# Model init and fitting
arima_base = ARIMA(wdf['Temp'], order=(5, 1, 0)).fit()
arima_exog = ARIMA(wdf['Temp'], exog=exog, order=(5, 1, 0)).fit()

# Compare Performance
print('=== No Exogenous Variable ===')
print('Mean Absolute Error: ', np.mean(np.abs(arima_base.resid[1:])))
print('Mean Squared Error: ', (arima_base.resid[1:] ** 2).mean())
print('Root Mean Squared Error: ', np.sqrt(arima_base.resid[1:] ** 2).mean())

print('\n=== With Exogenous Variable ===')
print('Mean Absolute Error: ', np.mean(np.abs(arima_exog.resid[1:])))
print('Mean Squared Error: ', (arima_exog.resid[1:] ** 2).mean())
print('Root Mean Squared Error: ', np.sqrt(arima_exog.resid[1:] ** 2).mean())
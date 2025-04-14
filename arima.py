import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Load dataset
df = pd.read_csv('/content/arima.csv')

# Parse dates
df['Date'] = pd.to_datetime(df['Date'])

# Group by date for national daily totals
daily_cases = df.groupby('Date')['Confirmed'].sum().sort_index()

# Ensure daily frequency
daily_cases = daily_cases.asfreq('D', method='ffill')

# ----- Check stationarity -----
print("\n📉 Augmented Dickey-Fuller Test:")
adf_result = adfuller(daily_cases)
print(f"ADF Statistic : {adf_result[0]:.4f}")
print(f"p-value       : {adf_result[1]:.4f}")

# Differencing if not stationary
if adf_result[1] > 0.05:
    print("⚠️ Series is not stationary. Applying first-order differencing.")
    daily_cases_diff = daily_cases.diff().dropna()
else:
    daily_cases_diff = daily_cases

# ----- Optional: Plot ACF & PACF to select p, d, q -----
# plot_acf(daily_cases_diff, lags=30)
# plot_pacf(daily_cases_diff, lags=30)
# plt.show()

# ----- Fit ARIMA model -----
# ARIMA(p, d, q) — we'll try (1,1,1) as a basic example
model = ARIMA(daily_cases, order=(1,1,1))
model_fit = model.fit()

# ----- Forecast future cases -----
forecast_steps = 30  # predict 30 days into the future
forecast = model_fit.get_forecast(steps=forecast_steps)
forecast_index = pd.date_range(start=daily_cases.index[-1] + pd.Timedelta(days=1), periods=forecast_steps)

# Confidence intervals
conf_int = forecast.conf_int()

# ----- Plot Actual vs Forecast -----
plt.figure(figsize=(12, 6))
plt.plot(daily_cases, label='Actual Data')
plt.plot(forecast_index, forecast.predicted_mean, label='Forecast', color='red')
plt.fill_between(forecast_index, 
                 conf_int['lower Confirmed'], 
                 conf_int['upper Confirmed'], color='pink', alpha=0.3)
plt.title('📈 ARIMA Forecast of COVID-19 Cases (Next 30 Days)')
plt.xlabel('Date')
plt.ylabel('Confirmed Cases')
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

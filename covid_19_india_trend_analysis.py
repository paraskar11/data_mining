import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

# Load dataset
df = pd.read_csv('/content/covid_19_india.csv')  # Use the actual path or name of your file

# Convert 'Date' to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Group by Date to get total daily confirmed cases across India
daily_cases = df.groupby('Date')['Confirmed'].sum()

# Sort by date just in case
daily_cases = daily_cases.sort_index()

# Fill missing dates if needed (optional)
daily_cases = daily_cases.asfreq('D', method='ffill')

# ----- Plot Time Series -----
plt.figure(figsize=(12, 5))
daily_cases.plot(title='📈 Daily Confirmed COVID-19 Cases in India')
plt.xlabel('Date')
plt.ylabel('Total Confirmed Cases')
plt.grid()
plt.tight_layout()
plt.show()

# ----- ADF Stationarity Test -----
print("\n📉 Augmented Dickey-Fuller Test:")
adf_result = adfuller(daily_cases)
print(f"ADF Statistic : {adf_result[0]:.4f}")
print(f"p-value       : {adf_result[1]:.4f}")
for key, value in adf_result[4].items():
    print(f"Critical Value ({key}): {value:.4f}")

if adf_result[1] <= 0.05:
    print("✅ Time series is stationary.")
else:
    print("❌ Time series is NOT stationary. Consider differencing or transformation.")

# ----- Decompose Time Series -----
decomposition = seasonal_decompose(daily_cases, model='additive', period=7)

# Plot the components
decomposition.plot()
plt.suptitle('📊 Time Series Decomposition (Trend + Seasonality + Residuals)', y=1.02)
plt.tight_layout()
plt.show()

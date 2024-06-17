import pandas as pd
import matplotlib.pyplot as plt
import pmdarima as pm

# Load the CSV file
df = pd.read_csv('downloads/time_series_15min_singleindex.csv')

# Select the column of interest and drop NaNs
df = df[['utc_timestamp', 'AT_solar_generation_actual']].dropna()

# Convert datetime
df['utc_timestamp'] = pd.to_datetime(df['utc_timestamp'])
df.set_index('utc_timestamp', inplace=True)

# Take every 48th value from the DataFrame
df_sampled = df.iloc[::48]

# Plot the sampled data to check
plt.plot(df_sampled['AT_solar_generation_actual'])
plt.show()
#or use the artificial data as 
#n_values = 1000
#period1 = 100  # Long period
#period2 = 10   # Short period

# Create time array
#t = np.arange(n_values)

# Create two periodic components
#component1 = (np.sin(2 * np.pi * t / period1)) * 2
#component2 = np.sin(2 * np.pi * t / period2)

# Combine the components
#combined_signal = component1 + component2

#time_index = pd.date_range(start='2000-01-01', periods=n_values, freq='D')
#series = pd.Series(combined_signal, index=time_index)
# Use pmdarima to find the best (p, d, q) parameters
#model = pm.auto_arima(df_sampled,start_p = 4,stop_p = 20, max_p = 1000,max_q = 1000, seasonal=True, stepwise=True, trace=True)
print(model.summary())
print(model)

#for a year we would need, (2*364) 
forecast = model.predict(n_periods=60*12,return_conf_int=False)

# Create a date range for the forecast
forecast_index = pd.date_range(start='1/1/2015', periods=60*12, freq='12H')


# Plot the original data and forecast
plt.figure(figsize=(14, 7))
#plt.plot(df_sampled['AT_solar_generation_actual'], label='Original Data')
plt.plot(forecast_index, forecast, linestyle='-', color='blue')
plt.legend()
plt.title('ARIMA Model for real-world data ')
plt.xlabel('Time')
plt.ylabel('position')

plt.savefig('downloads/arimareal.png')
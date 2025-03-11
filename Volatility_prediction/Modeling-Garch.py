import pandas as pd
import numpy as np
from arch import arch_model
import os

# Load data
portfolio = pd.read_csv(r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\portfolio\Portfolio_agg.csv')
window_size = 120  # 5 days
shift = 1  # shift by 1 hour

# Create separate lists to store the results of each model fit
egarch_results = []
gjrgarch_results = []
apgarch_results = []

# Create separate dataframes to store the forecasts of each model
egarch_forecasts = pd.DataFrame()
gjrgarch_forecasts = pd.DataFrame()
apgarch_forecasts = pd.DataFrame()

# Returns data
returns = portfolio['Returns']

# Function to forecast using fitted model
def check_residuals_and_forecast(garch_fit, horizon=6):
    np.random.seed(42)
    forecast_value = garch_fit.forecast(horizon=horizon, method='bootstrap')
    return forecast_value

# Loop through the data using the sliding window
for start in range(0, len(portfolio) - window_size + 1, shift):
    end = start + window_size
    window_data = returns.iloc[start:end]

    # Fit and forecast for each model separately
    for model_name in ['EGARCH', 'GJRGARCH', 'APGARCH']:
        # Initialize the model based on the model_name in each iteration
        if model_name == 'EGARCH':
            model = arch_model(window_data, vol='EGARCH', p=1, q=1)
        elif model_name == 'GJRGARCH':
            model = arch_model(window_data, vol='GARCH', p=1, q=1, o=1)
        elif model_name == 'APGARCH':
            model = arch_model(window_data, vol='APARCH', p=1, q=1)

        # Fit the model
        fit = model.fit(disp='off', options={'maxiter': 2000})

        # Forecast using the fitted model
        forecast_value = check_residuals_and_forecast(fit, horizon=6)
        variance_forecasts = forecast_value.variance[-1:]

        # Append forecast results to the appropriate dataframe and list
        if model_name == 'EGARCH':
            egarch_forecasts = pd.concat([egarch_forecasts, variance_forecasts], axis=0)
            egarch_results.append({
                'start': start,
                'end': end,
                'volatility': list(fit.conditional_volatility),
                'residuals': list(fit.resid),
                'params': fit.params
            })
        elif model_name == 'GJRGARCH':
            gjrgarch_forecasts = pd.concat([gjrgarch_forecasts, variance_forecasts], axis=0)
            gjrgarch_results.append({
                'start': start,
                'end': end,
                'volatility': list(fit.conditional_volatility),
                'residuals': list(fit.resid),
                'params': fit.params
            })
        elif model_name == 'APGARCH':
            apgarch_forecasts = pd.concat([apgarch_forecasts, variance_forecasts], axis=0)
            apgarch_results.append({
                'start': start,
                'end': end,
                'volatility': list(fit.conditional_volatility),
                'residuals': list(fit.resid),
                'params': fit.params
            })

# Convert results to DataFrames for analysis
egarch_results_df = pd.DataFrame(egarch_results)
gjrgarch_results_df = pd.DataFrame(gjrgarch_results)
apgarch_results_df = pd.DataFrame(apgarch_results)

# Save forecast DataFrames to CSV files
egarch_forecasts.reset_index(drop=False, inplace=True)
gjrgarch_forecasts.reset_index(drop=False, inplace=True)
apgarch_forecasts.reset_index(drop=False, inplace=True)

egarch_forecasts.to_csv(r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\Forecast_result\EGARCH_forecasts.csv', index=False)
gjrgarch_forecasts.to_csv(r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\Forecast_result\GJRGARCH_forecasts.csv', index=False)
apgarch_forecasts.to_csv(r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\Forecast_result\APGARCH_forecasts.csv', index=False)

# Save result DataFrames to CSV files
egarch_results_df.to_csv(r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\Forecast_result\egarch_results.csv', index=False)
gjrgarch_results_df.to_csv(r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\Forecast_result\gjrgarch_results.csv', index=False)
apgarch_results_df.to_csv(r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\Forecast_result\apgarch_results.csv', index=False)

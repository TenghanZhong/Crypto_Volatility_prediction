import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

avgarch_forecasts=pd.read_csv(r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\Forecast_result\avGARCH_forecasts.csv')
APgarch_forecasts=pd.read_csv(r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\Forecast_result\APGARCH_forecasts.csv')
figarch_forecasts=pd.read_csv(r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\Forecast_result\FIGARCH_forecasts.csv')
#avgarch_volume_forecasts=pd.read_csv(r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\Forecast_result\gjrGARCH_VOLUME_forecasts.csv')
#figarch_volume_forecasts=pd.read_csv(r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\Forecast_result\FIGARCH_VOLUME_forecasts.csv')

portfolio=pd.read_csv(r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\portfolio\Portfolio_agg.csv')

'''
def calculate_HAE(predicted_variance, realized_variance):
    """
    Calculate the HAE (Heteroscedastic Absolute Error) metric.

    Parameters:
    predicted_variance (float or array-like): The predicted variance values.
    realized_variance (float or array-like): The realized variance values.

    Returns:
    float or array-like: The calculated HAE values.
    """
    return abs(1 - (predicted_variance / realized_variance))


def calculate_ratio(predicted_variance,realized_variance):

    return predicted_variance / realized_variance

def calculate_MSE(predicted_variance, realized_variance):
    """
    Calculate the Mean Squared Error (MSE) between predicted and realized variances.

    Parameters:
    predicted_variance (float or array-like): The predicted variance values.
    realized_variance (float or array-like): The realized variance values.

    Returns:
    float: The calculated MSE value.
    """
    return np.mean((predicted_variance - realized_variance) ** 2)


def calculate_mean_errors(girgarch_forecasts, portfolio, calculate_HAE, calculate_MSE, forecast_name="Forecasts"):
    # Lists to store HAE and HSE values
    all_HAE = []
    all_MSE = []

    for i in range(len(girgarch_forecasts) - 6):
        index = girgarch_forecasts['index'][i]
        pred_val = girgarch_forecasts.iloc[i, 1:]
        realized_variance = portfolio.iloc[index + 1:index + 7]['Realized_Variance']

        HAE = [calculate_HAE(pred, real) for pred, real in zip(pred_val, realized_variance)]
        MSE = [calculate_MSE(pred, real) for pred, real in zip(pred_val, realized_variance)]

        # Append each calculation result for later analysis
        all_HAE.append(HAE)
        all_MSE.append(MSE)

    # Convert lists to NumPy arrays for easy averaging across columns
    mean_errors_HAE = np.mean(np.array(all_HAE), axis=0)
    mean_errors_MSE= np.mean(np.array(all_MSE), axis=0)

    # Print the results
    print(f"{forecast_name} HAE:",  mean_errors_HAE)
    print(f"{forecast_name} MSE:", mean_errors_MSE)

    return mean_errors_HAE, mean_errors_MSE


mean_errors_HAE, mean_errors_MSE=calculate_mean_errors(figarch_forecasts, portfolio, calculate_HAE, calculate_MSE,'figarch_forecasts')
#mean_errors_HAE, mean_errors_HSE=calculate_mean_errors(apgarch_forecasts, portfolio, calculate_HAE, calculate_MSE,'avgarch_forecasts')
mean_errors_HAE, mean_errors_HSE=calculate_mean_errors(avgarch_forecasts, portfolio, calculate_HAE, calculate_MSE,'figarch_volume_forecasts')


# Function to plot the comparison of realized variance with predicted variance for each lag, taking into account the offset
def plot_realized_vs_predicted_with_offset(forecasts, portfolio, num_lags=6, forecast_name='avgarch_forecasts',
                                           scale_factor=1e5):
    plt.figure(figsize=(15, 10))

    # Loop through each lag
    for lag in range(num_lags):
        plt.subplot(num_lags, 1, lag + 1)  # Create subplot for each lag

        realized_values = []
        predicted_values = []

        # Loop through each forecast and align it with the realized variance
        for i in range(len(forecasts) - num_lags):
            index = forecasts['index'][i]
            predicted_value = forecasts.iloc[i, lag + 1] * scale_factor  # Apply scale factor
            realized_value = portfolio.iloc[index + lag + 1]['Realized_Variance'] * scale_factor  # Apply scale factor

            # Append values for plotting
            realized_values.append(realized_value)
            predicted_values.append(predicted_value)

        # Plot realized and predicted variance for the current lag
        plt.plot(realized_values, label=f'Realized Variance (Lag {lag + 1})', color='blue')
        plt.plot(predicted_values, label=f'Predicted Variance (Lag {lag + 1})', color='orange')

        # Add titles and labels
        plt.title(f'{forecast_name} - Lag {lag + 1} Realized vs Predicted Variance')
        plt.xlabel('Time')
        plt.ylabel('Variance (Scaled)')
        plt.yscale('log')  # Use logarithmic scale for y-axis
        plt.legend()

    plt.tight_layout()
    plt.show()


# Call the plotting function for each forecast type with scaling applied
plot_realized_vs_predicted_with_offset(figarch_forecasts, portfolio, num_lags=6, forecast_name='figarch_forecasts',
                                       scale_factor=1e5)
plot_realized_vs_predicted_with_offset(avgarch_forecasts, portfolio, num_lags=6, forecast_name='apgarch_forecasts',
                                       scale_factor=1e5)
'''

#For test set comparision, compare the last 20%\
def plot_realized_vs_predicted_with_offset_sync(forecasts, portfolio, num_lags=6, forecast_name='avgarch_forecasts', scale_factor=1e5):
    plt.figure(figsize=(15, 10))

    # Loop through each lag
    for lag in range(num_lags):
        plt.subplot(num_lags, 1, lag + 1)  # Create subplot for each lag

        realized_values = []
        predicted_values = []

        # Loop through each row synchronously in forecasts and portfolio
        for i in range(len(forecasts) - num_lags):
            # Extract and scale the predicted and realized values for the current lag
            predicted_value = forecasts.iloc[i, lag + 1] * scale_factor  # Apply scale factor
            realized_value = portfolio.iloc[i + lag + 1]['Realized_Variance'] * scale_factor  # Apply scale factor

            # Append values for plotting
            realized_values.append(realized_value)
            predicted_values.append(predicted_value)

        # Plot realized and predicted variance for the current lag
        plt.plot(realized_values, label=f'Realized Variance (Lag {lag + 1})', color='blue')
        plt.plot(predicted_values, label=f'Predicted Variance (Lag {lag + 1})', color='orange')

        # Add titles and labels
        plt.title(f'{forecast_name} - Lag {lag + 1} Realized vs Predicted Variance')
        plt.xlabel('Time')
        plt.ylabel('Variance (Scaled)')
        plt.yscale('log')  # Use logarithmic scale for y-axis
        plt.legend()

    plt.tight_layout()
    plt.show()


def calculate_mean_errors_sync(girgarch_forecasts, portfolio, calculate_HAE, calculate_MSE, forecast_name="Forecasts"):
    # Lists to store HAE and MSE values
    all_HAE = []
    all_MSE = []

    # Loop through each row synchronously in girgarch_forecasts and portfolio
    for i in range(len(girgarch_forecasts) - 6):
        # Predicted values for the current row (assume columns 1 to 6 are the predictions)
        pred_val = girgarch_forecasts.iloc[i, 1:7]

        # Realized variance values for the next 6 timesteps in portfolio
        realized_variance = portfolio.iloc[i + 1:i + 7]['Realized_Variance']

        # Calculate HAE and MSE for each predicted-realized pair
        HAE = [calculate_HAE(pred, real) for pred, real in zip(pred_val, realized_variance)]
        MSE = [calculate_MSE(pred, real) for pred, real in zip(pred_val, realized_variance)]

        # Append each calculation result for later analysis
        all_HAE.append(HAE)
        all_MSE.append(MSE)

    # Convert lists to NumPy arrays for easy averaging across columns
    mean_errors_HAE = np.mean(np.array(all_HAE), axis=0)
    mean_errors_MSE = np.mean(np.array(all_MSE), axis=0)

    # Print the results
    print(f"{forecast_name} HAE:", mean_errors_HAE)
    print(f"{forecast_name} MSE:", mean_errors_MSE)

    return mean_errors_HAE, mean_errors_MSE



def calculate_and_plot_errors_sync_refined(forecasts, portfolio, forecast_name="Forecasts", num_lags=6, scale_factor=1e5):
    # Initialize lists to store errors for each lag
    mse_list = []
    mape_list = []

    # Loop through each lag
    for lag in range(num_lags):
        lag_mse = []
        lag_mape = []

        # Loop through each row synchronously in forecasts and portfolio
        for i in range(len(forecasts) - num_lags):
            # Extract and scale the predicted and realized values for the current lag
            predicted_value = forecasts.iloc[i, lag + 1] * scale_factor  # Apply scale factor
            realized_value = portfolio.iloc[i + lag + 1]['Realized_Variance'] * scale_factor  # Apply scale factor

            # Calculate MSE and MAPE
            mse = (predicted_value - realized_value) ** 2
            mape = np.abs((predicted_value - realized_value) / realized_value)

            # Append errors for plotting
            lag_mse.append(mse)
            lag_mape.append(mape)

        # Append lag errors to lists
        mse_list.append(lag_mse)
        mape_list.append(lag_mape)

    # Convert to DataFrame for box plotting
    mse_df = pd.DataFrame(mse_list).transpose()
    mape_df = pd.DataFrame(mape_list).transpose()

    # Plot box plots for MSE
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=mse_df, palette="Set2")
    plt.title(f'{forecast_name} - MSE for Lags (Scaled by {scale_factor})', fontsize=16)
    plt.xlabel('Lag', fontsize=14)
    plt.ylabel(f'MSE (Scaled by {scale_factor})', fontsize=14)
    plt.yscale('log')  # Use logarithmic scale for better visualization
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Plot box plots for MAPE
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=mape_df, palette="Set3")
    plt.title(f'{forecast_name} - MAPE for Lags (Scaled by {scale_factor})', fontsize=16)
    plt.xlabel('Lag', fontsize=14)
    plt.ylabel('MAPE', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    return mse_df, mape_df

# The refined function uses Seaborn for enhanced aesthetics, with clear titles, axis labels, and grid lines for better readability.
# Let me know if you'd like further customization!



# Split the dataset into training, validation, and test sets
train_size = int(len(portfolio) * 0.6)
val_size = int(len(portfolio) * 0.2)
p_test=portfolio[119:][train_size + val_size:]
av_test=avgarch_forecasts[train_size + val_size:]
fig_test=figarch_forecasts[train_size + val_size:]
ap_test=APgarch_forecasts[train_size + val_size:]

# Call the plotting function for each forecast type
plot_realized_vs_predicted_with_offset_sync(av_test, p_test, num_lags=6, forecast_name='avgarch_forecasts')
plot_realized_vs_predicted_with_offset_sync(fig_test, p_test, num_lags=6, forecast_name='figarch_forecasts')
plot_realized_vs_predicted_with_offset_sync(ap_test, p_test, num_lags=6, forecast_name='figarch_forecasts')

# Assuming 'portfolio', 'avgarch_forecasts', etc., are already defined as DataFrames
mse_df1, mape_df1=calculate_and_plot_errors_sync_refined(av_test, p_test, forecast_name='avgarch_forecasts', num_lags=6)
mse_df2, mape_df2=calculate_and_plot_errors_sync_refined(fig_test, p_test, forecast_name='figarch_forecasts', num_lags=6)
mse_df3, mape_df3=calculate_and_plot_errors_sync_refined(ap_test, p_test, forecast_name='APgarch_forecasts', num_lags=6)

# Define the directory path
save_directory = r"C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\error_series"

# Ensure the directory exists
os.makedirs(save_directory, exist_ok=True)

# Define file paths
mse_df1_path = os.path.join(save_directory, "mse_avgarch_forecasts.csv")
mape_df1_path = os.path.join(save_directory, "mape_avgarch_forecasts.csv")
mse_df2_path = os.path.join(save_directory, "mse_figarch_forecasts.csv")
mape_df2_path = os.path.join(save_directory, "mape_figarch_forecasts.csv")
mse_df3_path = os.path.join(save_directory, "mse_apgarch_forecasts.csv")
mape_df3_path = os.path.join(save_directory, "mape_apgarch_forecasts.csv")

# Save the DataFrames as CSV
mse_df1.to_csv(mse_df1_path, index=False)
mape_df1.to_csv(mape_df1_path, index=False)
mse_df2.to_csv(mse_df2_path, index=False)
mape_df2.to_csv(mape_df2_path, index=False)
mse_df3.to_csv(mse_df3_path, index=False)
mape_df3.to_csv(mape_df3_path, index=False)

print("DataFrames have been successfully saved!")

mean_errors_HAE_sync, mean_errors_HSE_sync=calculate_mean_errors_sync(av_test, p_test, calculate_HAE, calculate_MSE,'avgarch_forecasts')
mean_errors_HAE_sync, mean_errors_HSE_sync=calculate_mean_errors_sync(fig_test, p_test, calculate_HAE, calculate_MSE,'figgarch_forecasts')





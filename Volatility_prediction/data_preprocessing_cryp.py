import pandas as pd
import numpy as np
import os

# Folder paths
folder_path = r"C:\Users\26876\Desktop\lstmgarch_project\Original_crydata"
folder_path2 = r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\Agg_data'

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # Check if it's a file
    if os.path.isfile(file_path):
        try:
            # Read CSV data
            data = pd.read_csv(file_path)

            # Ensure 'DateTime' column is in datetime format and set as index
            data['DateTime'] = pd.to_datetime(data['DateTime'])
            data.set_index('DateTime', inplace=True)

            # Calculate logarithmic return and volume
            data['logarithmic_return'] = np.log(data['Close']).diff().fillna(0)
            data['Volume'] = np.log(data['Volume'])

            # Calculate realized variance (5-minute squared log returns)
            data['logarithmic_return_squared'] = data['logarithmic_return'] ** 2

            # Aggregate to hourly data
            hourly_realized_variance = data['logarithmic_return_squared'].resample('1H').sum()
            hourly_volume = data['Volume'].resample('1H').sum()

            Hourly_return=data.groupby(data.index.floor('H')).apply(
                lambda x:np.log(x['Close'].iloc[-1]) - np.log(x['Close'].iloc[0]))

            # Final result with realized variance
            result = pd.DataFrame({
                'hourly_sum_volume': hourly_volume,
                'realized_variance': hourly_realized_variance,
                'Hourly_return': Hourly_return
            })
            result['hourly_sum_volume'].replace(0, np.nan, inplace=True)

            result.interpolate(method='linear',inplace=True)

            # Save result to CSV
            result_file_path = os.path.join(folder_path2, f'{filename[:-9]}_agg.csv')
            result.to_csv(result_file_path)
            print(f'{filename[:-9]} complete!')

        except PermissionError:
            print(f"Permission denied: Could not write to file {result_file_path}")
        except Exception as e:
            print(f"An error occurred with file {filename}: {e}")



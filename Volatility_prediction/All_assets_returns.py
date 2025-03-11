import pandas as pd
import numpy as np
import os

folder_path = r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata'
All_assets_return = pd.DataFrame()

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)

    # Check if it's a file
    if os.path.isfile(file_path):
        try:
            # Read CSV data
            data = pd.read_csv(file_path)

            # Ensure 'DateTime' column is in datetime format and set as index
            if 'DateTime' in data.columns:
                data['DateTime'] = pd.to_datetime(data['DateTime'])
                data.set_index('DateTime', inplace=True)
            else:
                raise ValueError(f"'DateTime' column not found in {filename}")

            # Generate a complete 5-minute interval index for the date range
            complete_index = pd.date_range(start=data.index.min(), end=data.index.max(), freq='5T')

            # Reindex the data to include all 5-minute intervals, filling with NaN where missing
            data = data.reindex(complete_index)

            # Interpolate missing values linearly
            data['Close'].interpolate(method='linear', inplace=True)

            # Calculate logarithmic return
            data['logarithmic_return'] = np.log(data['Close']).diff().fillna(0)

            # Add to the DataFrame using the filename without extension
            asset_name = os.path.splitext(filename)[0]
            All_assets_return = All_assets_return.join(data['logarithmic_return'].rename(asset_name), how='outer')

        except ValueError as ve:
            print(f"ValueError with file {filename}: {ve}")
        except Exception as e:
            print(f"An error occurred with file {filename}: {e}")

# Check result
print(All_assets_return.head())

All_assets_return.to_csv(r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\portfolio\All_assets_return.csv')


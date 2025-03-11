import pandas as pd
import numpy as np
import os

realized_variance=[]
All_returns=pd.read_csv(r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\portfolio\All_assets_return.csv')
All_returns.set_index('Unnamed: 0',inplace=True)
All_returns.index=pd.to_datetime(All_returns.index)

# Convert all columns to numeric, coercing errors to NaN
All_returns = All_returns.apply(pd.to_numeric, errors='coerce')

# List of asset column names
columns = All_returns.columns.tolist()

# Initialize list to store realized variance for each hour
realized_variance = []

# Group the data by each hour
grouped = All_returns.groupby(All_returns.index.floor('H'))

# Loop through each hourly group
for hour, group in grouped:
    sum_value = 0
    for i in columns:
        for j in columns:
            # Sum of the product of each 5-minute return
            sum_value += (group[i] * group[j]).sum()

    # Append the realized variance for the hour, divided by 100 (assuming scaling is needed)
    realized_variance.append({'Hour': hour, 'Realized_Variance': sum_value / 100})
# Convert to DataFrame
realized_variance_df = pd.DataFrame(realized_variance)

# Convert the 'Hour' column to datetime if needed
realized_variance_df['Hour'] = pd.to_datetime(realized_variance_df['Hour'])


# Portfolio
folder_path = r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\Agg_data'
portfolio = realized_variance_df.reset_index('Hour').copy()

returns = pd.DataFrame()
Volume = pd.DataFrame()

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    data = pd.read_csv(file_path)
    if data.isnull().sum().sum() >= 1:
        print(f'{filename} 有空值，位置在{np.where(data.isnull())}')
    returns[filename] = data['Hourly_return']
    Volume[filename] = data['hourly_sum_volume']

# 计算组合的平均收益和成交量
portfolio['Returns'] = returns.mean(axis=1).tolist()
portfolio['Volume'] = Volume.mean(axis=1).tolist()

portfolio.reset_index(drop=False,inplace=True)

# 保存组合数据到 CSV 文件
folder_path2 = r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\portfolio'
portfolio_file_path = os.path.join(folder_path2, 'Portfolio_agg.csv')
portfolio.to_csv(portfolio_file_path, index=False)


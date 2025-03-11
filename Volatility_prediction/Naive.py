import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

egarch_forecasts=pd.read_csv(r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\Forecast_result\FIGARCH_forecasts.csv')
portfolio=pd.read_csv(r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\portfolio\Portfolio_agg.csv')
naive=portfolio.copy()

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
def calculate_MSE(predicted_variance, realized_variance):
    """
    Calculate the MSE (Mean Squared Error) metric.
    """
    return (predicted_variance - realized_variance) ** 2

def calculate_ratio(predicted_variance,realized_variance):

    return abs(predicted_variance / realized_variance)

# Lists to store HAE and HSE values
all_HAE = []
all_HSE = []
all_abs= []

for i in range(len(naive)-78):
    index= 72+i
    pred_val=[]
    for j in range(0,6):
        pred_val.append( naive[ i+j :index + j]['Realized_Variance'].mean()   )

    realized_variance=naive.iloc[index:index+6]['Realized_Variance'].values
    HAE=[ calculate_HAE(pred,real) for pred,real in zip(pred_val,realized_variance)]
    HSE=[ calculate_MSE(pred,real) for pred,real in zip(pred_val,realized_variance)]

    # Append each calculation result for later analysis
    all_HAE.append(HAE)
    all_HSE.append(HSE)


# Convert to a NumPy array for easy averaging across columns
error_array = np.array(all_HAE)

# Calculate the mean for each hour (column-wise)
mean_errors_HAE = np.mean(error_array, axis=0)

# Convert to a NumPy array for easy averaging across columns
error_array = np.array(all_HSE)

# Calculate the mean for each hour (column-wise)
mean_errors_MSE = np.mean(error_array, axis=0)

print(mean_errors_HAE )
print(mean_errors_MSE )

'''
naive:HAE
array([ 2.37257207,  2.31823075, 69.86878869,  2.29517883,  2.3001328 ,
        2.84512202,  2.69858912,  2.35641661,  1.49105191,  1.02084703,
        1.10968928,  1.09898717,  1.24676144,  1.33466454,  1.68241888,
        2.01930083,  2.24177255,  2.16220238,  2.52285277,  2.62060856,
        1.36715168,  1.43184131,  1.8080397 ,  2.48302783])
        
HAE,FIG:mean_errors_HAE
mean_errors_HAE
array([ 2.56260676,  2.80815154, 28.62539971,  2.48273756,  2.455657  ,
        2.62684997,  2.8719261 ,  2.61527489,  1.96180784,  2.01505126,
        1.52931373,  1.53831323,  1.43372873,  1.75833369,  4.61755547,
        4.06702984,  3.44729283,  3.19673639,  3.67872877,  3.93006795,
        1.89954769,  1.92534651,  2.87788864,  3.35925703])
'''

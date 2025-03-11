import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import GRU
from keras.layers import Dropout
import tensorflow as tf
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
import os
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.losses import Huber
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Layer, Conv1D, MaxPooling1D, Dense, GRU, Dropout, Multiply
from keras import backend as K
from keras.models import Sequential
import tensorflow as tf
import json
from keras.models import load_model
import seaborn as sns

def expand_volatility_residuals(result_data):
    # Extracting volatility and residuals by converting them from strings to lists
    volatility_expanded = pd.DataFrame(result_data['volatility'].apply(eval).tolist())
    residuals_expanded = pd.DataFrame(result_data['residuals'].apply(eval).tolist())

    # Renaming columns for clarity
    volatility_expanded.columns = [f'volatility_{i}' for i in range(volatility_expanded.shape[1])]
    residuals_expanded.columns = [f'residual_{i}' for i in range(residuals_expanded.shape[1])]

    # Process 'params' to extract parameter values into separate columns
    params_expanded = result_data['params'].astype(str).str.split('\n', expand=True)
    params_expanded = params_expanded.applymap(lambda x: float(x.split()[-1]) if isinstance(x, str) and x.split()[-1].replace('.', '', 1).replace('-', '', 1).isdigit() else np.nan)
    params_expanded.columns = params_expanded.iloc[0].apply(lambda x: x.split()[0] if isinstance(x, str) else '').tolist()

    # Combine expanded columns with the original start and end columns
    expanded_result = pd.concat([result_data[['start', 'end']], volatility_expanded, residuals_expanded, params_expanded], axis=1)

    return expanded_result, volatility_expanded.values, residuals_expanded.values


def create_sequences_volatility(data, timesteps, prediction_length, shift, volatility, residuals):
    X, y = [], []

    for i in range(timesteps, len(data) - prediction_length, shift):
        # Extract the main data sequence
        main_sequence = data[(i - timesteps):i, :]

        # Extract the corresponding 120-timestep sequences for volatility and residuals
        volatility_sequence = volatility[i - timesteps].reshape(timesteps, -1)  # Ensure 2D shape
        residuals_sequence = residuals[i - timesteps].reshape(timesteps, -1)    # Ensure 2D shape

        # Debugging prints to check shapes and sample values
        print(f"main_sequence shape: {main_sequence.shape}")
        print(f"volatility_sequence shape: {volatility_sequence.shape}")
        print(f"residuals_sequence shape: {residuals_sequence.shape}")

        # Check if shapes are aligned for concatenation
        assert main_sequence.shape[0] == timesteps, "Mismatch in timesteps for main_sequence."
        assert volatility_sequence.shape[0] == timesteps, "Mismatch in timesteps for volatility_sequence."
        assert residuals_sequence.shape[0] == timesteps, "Mismatch in timesteps for residuals_sequence."

        # Concatenate the main sequence with volatility and residuals sequences
        combined_sequence = np.concatenate([main_sequence, volatility_sequence, residuals_sequence], axis=1)

        # Additional debug statement to check combined sequence
        print(f"combined_sequence shape: {combined_sequence.shape}")
        print(f"Sample combined_sequence values:\n{combined_sequence[:2]}")  # Print first two rows for verification

        # Append the combined sequence to X
        X.append(combined_sequence)

        # Define y as the target prediction sequence (first feature)
        y.append(data[i:i + prediction_length, 0])  # Predicting the first feature

    X = np.array(X)
    y = np.array(y)

    return X, y

def create_sequences(data, timesteps, prediction_length, shift):
    X, y = [], []
    for i in range(timesteps, len(data) - prediction_length, shift):
        X.append(data[(i - timesteps):i, :])
        y.append(data[i:i + prediction_length, 0])  # Predicting the first feature
    X = np.array(X)
    y = np.array(y)
    return X, y

def preprocess_data(dataset_train, timesteps, prediction_length, shift, objective, volatility, residuals):
    # Convert 'Hour' column to a datetime format and sort by it
    dataset_train['Hour'] = pd.to_datetime(dataset_train['Hour'].astype(str)).dt.strftime('%Y%m%d')
    dataset_train = dataset_train.sort_values(by='Hour', ascending=True).reset_index(drop=True)

    # Only select columns specified in the objective
    train_set = dataset_train[list(objective)].values

    # Define sizes for validation, training, and test sets
    val_size = int(len(train_set) * 0.2)
    train_size = int(len(train_set) * 0.6)

    # Define validation, training, and test sets
    val_data = train_set[:val_size]
    train_data = train_set[val_size:val_size + train_size]
    test_data = train_set[val_size + train_size:]

    # Scale the main data based on training data only
    scaler_main =StandardScaler()
    train_data_scaled = scaler_main.fit_transform(train_data)
    val_data_scaled = scaler_main.transform(val_data)
    test_data_scaled = scaler_main.transform(test_data)

    # Scale the volatility and residuals independently based on their respective training parts
    scaler_volatility = MinMaxScaler()
    scaler_residuals = MinMaxScaler()
    volatility_train_scaled = scaler_volatility.fit_transform(volatility[:train_size])
    residuals_train_scaled = scaler_residuals.fit_transform(residuals[:train_size])

    # Scale validation and test sets for volatility and residuals
    volatility_val_scaled = scaler_volatility.transform(volatility[train_size:train_size + val_size])
    volatility_test_scaled = scaler_volatility.transform(volatility[train_size + val_size:])

    residuals_val_scaled = scaler_residuals.transform(residuals[train_size:train_size + val_size])
    residuals_test_scaled = scaler_residuals.transform(residuals[train_size + val_size:])

    # Create sequences for each set
    X_train, y_train = create_sequences_volatility(train_data_scaled, timesteps, prediction_length, shift, volatility_train_scaled, residuals_train_scaled)
    X_val, y_val = create_sequences_volatility(val_data_scaled, timesteps, prediction_length, shift, volatility_val_scaled, residuals_val_scaled)
    X_test, y_test = create_sequences_volatility(test_data_scaled, timesteps, prediction_length, shift, volatility_test_scaled, residuals_test_scaled)

    # Number of features (indicators)
    indicators = X_train.shape[2]

    return X_train, y_train, X_val, y_val, X_test, y_test, indicators, scaler_main, scaler_volatility, scaler_residuals


def build_GRU_model(X_train, y_train,indicators,timesteps,prediction_length,filters, kernel_size
    , units, dropout_rate,epochs,batch_size):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 若设置为‘-1’则是使用CPU，忽略GPU
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices(
        'GPU')))  # 如果输出是 Num GPUs Available: 0，这意味着 TensorFlow 将只使用 CPU 运行。

    regressor = Sequential()


    # CNN layer: 1D Convolution followed by MaxPooling
    '''
    regressor.add(
        Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(X_train.shape[1], indicators)))
    regressor.add(MaxPooling1D(pool_size=2))
    regressor.add(Dropout(rate=dropout_rate))
    '''


    input_shape = (timesteps, indicators)
    #X_train.shape[1]=timesteps
    # 第一层 GRU
    regressor.add(GRU(units=units, activation='tanh', return_sequences=True, input_shape=input_shape))
    regressor.add(Dropout(rate=dropout_rate))

    regressor.add(GRU(units=units, activation='tanh', return_sequences=True, input_shape=input_shape))
    regressor.add(Dropout(rate=dropout_rate))

    regressor.add(GRU(units=units, activation='tanh', return_sequences=False))
    regressor.add(Dropout(rate=dropout_rate))

    # Apply custom Attention layer
    #regressor.add(AttentionLayer())

    # 输出层
    regressor.add(Dense(units=prediction_length))  # 预测一个值
    # 编译模型
    regressor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='huber_loss')

    # loss='mean_squared_error'因为预测的是股票价格，是一个数值，类似于回归问题，就使用mse。（分类问题用cross——entropy）
    # 这里不用,metrics=['accuracy']，也是因为预测的是股票价格（而不是分类是否正确的问题）
    regressor.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    return regressor


def make_predictions_all(X_test, y_test, model, scaler):
    # Make predictions directly with preprocessed X_test
    predictions = model.predict(X_test)

    # first_values = predictions[:, 0]
    first_values = predictions[:, 0].flatten()

    # Flatten y_test to match the structure
    y_test = y_test.flatten()

    # Create dummy arrays with the same number of columns as the original dataset
    dummy_predicted = np.zeros((len(first_values), scaler.n_features_in_))
    dummy_y_test = np.zeros((len(y_test), scaler.n_features_in_))

    # Fill the first column with the first predicted values and actual target values
    dummy_predicted[:, 0] = first_values
    dummy_y_test[:, 0] = y_test

    # Inverse transform both predicted and actual values
    predicted_values_inversed = scaler.inverse_transform(dummy_predicted)[:, 0]
    y_test_inversed = scaler.inverse_transform(dummy_y_test)[:, 0]

    return y_test_inversed, predicted_values_inversed


def make_predictions_separate(X_test, y_test, model, scaler, prediction_length=6):
    # Make predictions directly with preprocessed X_test
    predictions = model.predict(X_test)

    # Check if predictions match the expected shape
    if predictions.shape[1] != prediction_length:
        raise ValueError(f"Expected predictions to have {prediction_length} columns, but got {predictions.shape[1]}")

    # Initialize lists to store individual series for each prediction hour
    predicted_series_list = [[] for _ in range(prediction_length)]
    y_test_series_list = [[] for _ in range(prediction_length)]

    # Iterate through predictions and true values to separate them by prediction step
    for i in range(len(predictions)):
        for j in range(prediction_length):
            predicted_series_list[j].append(predictions[i, j])
            y_test_series_list[j].append(y_test[i, j])

    # Create dummy arrays to match the scaler's expected input shape
    predicted_series_inversed = []
    y_test_series_inversed = []
    for i in range(prediction_length):
        dummy_predicted = np.zeros((len(predicted_series_list[i]), scaler.n_features_in_))
        dummy_y_test = np.zeros((len(y_test_series_list[i]), scaler.n_features_in_))

        # Fill the first column with the actual predicted values and actual target values
        dummy_predicted[:, 0] = predicted_series_list[i]
        dummy_y_test[:, 0] = y_test_series_list[i]

        # Inverse transform both predicted and actual values
        predicted_values_inversed = scaler.inverse_transform(dummy_predicted)[:, 0]
        y_test_values_inversed = scaler.inverse_transform(dummy_y_test)[:, 0]

        # Append the inversed values to the respective lists
        predicted_series_inversed.append(predicted_values_inversed)
        y_test_series_inversed.append(y_test_values_inversed)

    # Create a DataFrame to store real values and their corresponding predicted values for each time step
    dataframes = []
    for i in range(prediction_length):
        df = pd.DataFrame({
            f'Real_Value_Timestep_{i+1}': y_test_series_inversed[i],
            f'Predicted_Value_Timestep_{i+1}': predicted_series_inversed[i]
        })
        dataframes.append(df)

    # Concatenate all DataFrames along columns
    final_dataframe = pd.concat(dataframes, axis=1)

    return final_dataframe

def calibration_report_MSE_sep(final_dataframe, prediction_length=6, save_path=None,name="gru_avagrch", scale_factor=1e5):
    """
    Calculates Mean Squared Error (MSE) between real and predicted values for each prediction timestep,
    and saves the full MSE series as a DataFrame if a save path is provided.
    """
    mse_list = []
    mse_series = {}

    for i in range(prediction_length):
        real_price = final_dataframe[f'Real_Value_Timestep_{i+1}'].values.flatten()
        predicted_price = final_dataframe[f'Predicted_Value_Timestep_{i+1}'].values.flatten()

        if len(real_price) != len(predicted_price):
            raise ValueError("The lengths of real_price and predicted_price must be the same")

        # Calculate Mean Squared Error (MSE)
        mse = np.mean((real_price - predicted_price) ** 2)
        mse_series[f'Timestep_{i+1}'] = (real_price*scale_factor - predicted_price*scale_factor) ** 2   # Save the full series for this timestep
        mse_list.append(mse)

        # Print error summary for each timestep
        print(f"Mean Squared Error (MSE) for Timestep {i+1}: {mse:.9f}")

    # Convert MSE series dictionary to a DataFrame
    mse_df = pd.DataFrame(mse_series)

    # Save the DataFrame as a CSV file if a save path is provided
    if save_path:
        mse_file_path = os.path.join(save_path, f'{name}_mse_series.csv')
        mse_df.to_csv(mse_file_path, index=False)
        print(f"MSE series DataFrame saved to {mse_file_path}")

    return mse_list, mse_df

def calibration_report_MAPE_sep(final_dataframe, prediction_length=6, save_path=None,name="gru_avagrch"):
    """
    Calculates Mean Absolute Percentage Error (MAPE) between real and predicted values for each prediction timestep,
    and saves the full MAPE series as a DataFrame if a save path is provided.
    """
    mape_list = []
    mape_series = {}

    for i in range(prediction_length):
        real_price = final_dataframe[f'Real_Value_Timestep_{i+1}'].values.flatten()
        predicted_price = final_dataframe[f'Predicted_Value_Timestep_{i+1}'].values.flatten()

        if len(real_price) != len(predicted_price):
            raise ValueError("The lengths of real_price and predicted_price must be the same")

        # Calculate Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((real_price - predicted_price) / real_price))
        mape_series[f'Timestep_{i+1}'] = np.abs((real_price - predicted_price) / real_price)   # Save the full series
        mape_list.append(mape)

        # Print error summary for each timestep
        print(f"Mean Absolute Percentage Error (MAPE) for Timestep {i+1}: {mape:.9f}%")

    # Convert MAPE series dictionary to a DataFrame
    mape_df = pd.DataFrame(mape_series)

    # Save the DataFrame as a CSV file if a save path is provided
    if save_path:
        mape_file_path = os.path.join(save_path, f'{name}_mape_series.csv')
        mape_df.to_csv(mape_file_path, index=False)
        print(f"MAPE series DataFrame saved to {mape_file_path}")

    return mape_list, mape_df


def preprocess_data_pure(dataset_train, timesteps, prediction_length, shift, objective):
    # Convert 'Hour' column to a datetime format and sort by it
    dataset_train['Hour'] = pd.to_datetime(dataset_train['Hour'].astype(str)).dt.strftime('%Y%m%d')
    dataset_train = dataset_train.sort_values(by='Hour', ascending=True).reset_index(drop=True)

    # Only select columns specified in the objective
    train_set = dataset_train[[objective]].values

    # Number of features (indicators)
    indicators = train_set.shape[1]

    # Define sizes for validation, training, and test sets
    val_size = int(len(train_set) * 0.2)
    train_size = int(len(train_set) * 0.6)

    # Define validation, training, and test sets
    val_data = train_set[:val_size]
    train_data = train_set[val_size:val_size + train_size]
    test_data = train_set[val_size + train_size:]

    # Scale only based on training data
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)

    # Transform validation and test data with the training scaler
    val_data_scaled = scaler.transform(val_data)
    test_data_scaled = scaler.transform(test_data)

    # Create sequences for each set
    X_train, y_train = create_sequences(train_data_scaled, timesteps, prediction_length, shift)
    X_val, y_val = create_sequences(val_data_scaled, timesteps, prediction_length, shift)
    X_test, y_test = create_sequences(test_data_scaled, timesteps, prediction_length, shift)

    return X_train, y_train, X_val, y_val, X_test, y_test, indicators, scaler

def preprocess_data_mix(dataset_train, timesteps, prediction_length, shift, objective):
    # Convert 'Hour' column to a datetime format and sort by it
    dataset_train['Hour'] = pd.to_datetime(dataset_train['Hour'].astype(str)).dt.strftime('%Y%m%d')
    dataset_train = dataset_train.sort_values(by='Hour', ascending=True).reset_index(drop=True)

    # Only select columns specified in the objective
    train_set = dataset_train[list(objective)].values

    # Number of features (indicators)
    indicators = train_set.shape[1]

    # Define sizes for validation, training, and test sets
    val_size = int(len(train_set) * 0.2)
    train_size = int(len(train_set) * 0.6)

    # Define validation, training, and test sets
    val_data = train_set[:val_size]
    train_data = train_set[val_size:val_size + train_size]
    test_data = train_set[val_size + train_size:]

    # Scale only based on training data
    scaler = StandardScaler()
    train_data_scaled = scaler.fit_transform(train_data)

    # Transform validation and test data with the training scaler
    val_data_scaled = scaler.transform(val_data)
    test_data_scaled = scaler.transform(test_data)

    # Create sequences for each set
    X_train, y_train = create_sequences(train_data_scaled, timesteps, prediction_length, shift)
    X_val, y_val = create_sequences(val_data_scaled, timesteps, prediction_length, shift)
    X_test, y_test = create_sequences(test_data_scaled, timesteps, prediction_length, shift)

    return X_train, y_train, X_val, y_val, X_test, y_test, indicators, scaler

def plot_error_series_boxplots(error_series, prediction_length, title, ylabel, save_path=None, scale="linear"):
    """
    Plots box plots for error series across prediction timesteps with dynamic y-limits.
    """
    # Create DataFrame for plotting
    error_df = pd.DataFrame(error_series)

    # Rename columns to represent timesteps
    error_df.columns = [f'Timestep_{i + 1}' for i in range(prediction_length)]

    # Automatically determine y-limits based on data
    min_val = error_df.min().min()  # Find the minimum value across all timesteps
    max_val = error_df.max().max()  # Find the maximum value across all timesteps
    padding = 0.1 * (max_val - min_val)  # Add 10% padding for visualization
    y_limits = (max(min_val - padding, 0), max_val + padding)

    # Create the plot
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=error_df, palette="Set2")
    plt.title(title, fontsize=16)
    plt.xlabel('Prediction Timesteps', fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.yscale(scale)
    plt.ylim(y_limits)  # Set dynamic y-limits
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path)
        print(f"Boxplot saved to {save_path}")

    plt.show()



if __name__ == "__main__":
    portfolio = pd.read_csv(r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\portfolio\Portfolio_agg.csv')
    result = pd.read_csv(
        r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\Forecast_result\avgarch_results.csv')
    result2 = pd.read_csv(
        r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\Forecast_result\figarch_results.csv')

    expanded_result, volatility, residuals = expand_volatility_residuals(result)
    expanded_resultfig, volatilityfig, residualsfig = expand_volatility_residuals(result2)

    # 分阶段优化
    objective = ('Realized_Variance', 'Volume')  # 要预测的值为第一个
    objective_single = ('Realized_Variance')
    timesteps = 120  # 单次训练步长
    shift= 1  #rolling shift
    prediction_length = 6 #单次预测步长
    datasets = portfolio.copy()
    best_params = None
    combo_count = 0

    # Replace infinite values with NaN
    datasets.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Interpolate NaN values (linear interpolation by default)
    datasets.interpolate(method='linear', inplace=True)

    (X_train, y_train, X_val, y_val, X_test, y_test, indicators, sc,
     scaler_volatility, scaler_residuals) = preprocess_data(datasets,
                                                            timesteps, prediction_length, shift, objective, volatility,
                                                            residuals)
    (X_trainfig, y_trainfig, X_valfig, y_valfig, X_testfig, y_testfig, indicatorsfig, scfig,
     scaler_volatilityfig, scaler_residualsfig) = preprocess_data(datasets,
                                                            timesteps, prediction_length, shift, objective, volatilityfig,
                                                            residualsfig)

    X_train_gru, y_train_gru, X_val_gru, y_val_gru, X_test_gru, y_test_gru, indicators_gru, sc_gru = preprocess_data_pure(datasets,
                                                                                     timesteps, prediction_length,
                                                                                     shift, objective_single)
    X_train_gru_vol, y_train_gru_vol, X_val_gru_vol, y_val_gru_vol, X_test_gru_vol, y_test_gru_vol, indicators_gru_vol, sc_gru_vol =\
        preprocess_data_mix(datasets,timesteps, prediction_length,shift, objective)



    # Define the path to the saved GRU model
    gru_avgarch_model = r"C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\GRU_family_result\gru_avgarch_model.h5"
    gru_figgarch_model = r"C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\GRU_family_result\gru_figgarch_model.h5"
    gru_model = r"C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\GRU_family_result\gru_model.h5"
    gru_volume_model = r"C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\GRU_family_result\gru_volume_model.h5"

    gru_avgarch_model = load_model(gru_avgarch_model)
    gru_figgarch_model = load_model(gru_figgarch_model)
    gru_model = load_model(gru_model)
    gru_volume_model = load_model(gru_volume_model)


    #Best parameters: {'activation': 'tanh', 'batch_size': 16, 'dropout_rate': 0.2, 'epochs': 25, 'layer_numbers': 3, 'units': 256}
    #Best MAE: 0.33480495010882055
    '''
    # 得到predicted_error 绘制对比图--------only need the first value of prediction,since shift=1,predict_length=6
    real_price, predicted_price = make_predictions_all (X_test,y_test,regressor,sc)
    plot_predictions(real_price, predicted_price)
    print(calibration_report_MAPE(real_price, predicted_price))
    print(calibration_report_MSE(real_price, predicted_price))
    '''

    data_comparision=make_predictions_separate(X_test, y_test,gru_avgarch_model,sc, prediction_length)
    data_comparisionfig = make_predictions_separate(X_testfig, y_testfig, gru_figgarch_model, scfig, prediction_length)
    data_comparision_gru = make_predictions_separate(X_test_gru, y_test_gru, gru_model, sc_gru, prediction_length)
    data_comparision_gru_vol = make_predictions_separate(X_test_gru_vol, y_test_gru_vol,gru_volume_model, sc_gru_vol, prediction_length)


    save_path = r"C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\error_series"
    mape_list,mape_df=calibration_report_MAPE_sep(data_comparision, prediction_length,save_path,name="gru_avagrch_mape")
    print(mape_list)
    mse_list,mse_df=calibration_report_MSE_sep(data_comparision, prediction_length,save_path,name="gru_avagrch_mse")
    print(mse_list)
    mape_listfig, mape_dffig = calibration_report_MAPE_sep(data_comparisionfig, prediction_length, save_path, name="gru_figagrch_mape")
    print(mape_list)
    mse_listfig, mse_dffig = calibration_report_MSE_sep(data_comparisionfig, prediction_length, save_path, name="gru_figagrch_mse")
    print(mse_list)
    mape_list_gru, mape_df_gru = calibration_report_MAPE_sep(data_comparision_gru, prediction_length, save_path,
                                                           name="gru_mape")
    print(mape_list)
    mse_list_gru, mse_df_gru = calibration_report_MSE_sep(data_comparision_gru, prediction_length, save_path,
                                                        name="gru_mse")
    print(mse_list)
    mape_list_gru_vol, mape_df_gru_vol = calibration_report_MAPE_sep(data_comparision_gru_vol, prediction_length, save_path,
                                                             name="gru_vol_mape")
    print(mape_list)
    mse_list_gru_vol, mse_df_gru_vol = calibration_report_MSE_sep(data_comparision_gru_vol, prediction_length, save_path,
                                                          name="gru_vol_mse")
    print(mse_list)

    output_path = r"C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\error_series\Box_plot"
    # Create directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)

    # Plot boxplots for each model's MSE and MAPE
    plot_error_series_boxplots(
        mse_df, prediction_length, "MSE Boxplot - GRU_AVGARCH", "MSE",
        save_path=os.path.join(output_path, "gru_avgarch_mse_boxplot.png"),
        scale="linear"
    )

    plot_error_series_boxplots(
        mape_df, prediction_length, "MAPE Boxplot - GRU_AVGARCH", "MAPE",
        save_path=os.path.join(output_path, "gru_avgarch_mape_boxplot.png"),
        scale="log"
    )

    plot_error_series_boxplots(
        mse_dffig, prediction_length, "MSE Boxplot - GRU_FIGARCH", "MSE",
        save_path=os.path.join(output_path, "gru_figarch_mse_boxplot.png"),
        scale="linear"
    )

    plot_error_series_boxplots(
        mape_dffig, prediction_length, "MAPE Boxplot - GRU_FIGARCH", "MAPE",
        save_path=os.path.join(output_path, "gru_figarch_mape_boxplot.png"),
        scale="log"
    )

    plot_error_series_boxplots(
        mse_df_gru, prediction_length, "MSE Boxplot - GRU", "MSE",
        save_path=os.path.join(output_path, "gru_mse_boxplot.png"),
        scale="linear"
    )

    plot_error_series_boxplots(
        mape_df_gru, prediction_length, "MAPE Boxplot - GRU", "MAPE",
        save_path=os.path.join(output_path, "gru_mape_boxplot.png"),
        scale="log"
    )

    plot_error_series_boxplots(
        mse_df_gru_vol, prediction_length, "MSE Boxplot - GRU_VOLUME", "MSE",
        save_path=os.path.join(output_path, "gru_volume_mse_boxplot.png"),
        scale="linear"
    )

    plot_error_series_boxplots(
        mape_df_gru_vol, prediction_length, "MAPE Boxplot - GRU_VOLUME", "MAPE",
        save_path=os.path.join(output_path, "gru_volume_mape_boxplot.png"),
        scale="log"
    )






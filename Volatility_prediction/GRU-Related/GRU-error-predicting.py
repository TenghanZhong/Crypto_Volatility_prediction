import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
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



def extract_error(figarch_forecasts, portfolio):
    error_list = []
    realized_variance_list=[]
    pre_list=[]

    for i in range(len(figarch_forecasts) - 6):
        index = figarch_forecasts['index'][i]
        pred_val = figarch_forecasts.iloc[i, 1:]
        realized_variance = portfolio.iloc[index + 1:index + 7]['Realized_Variance']

        error = (pred_val.values - realized_variance.values).tolist()
        error_list.append({'index': index, 'error': error})
        realized_variance_list.append({'index': index, 'realized_variance': realized_variance.values.tolist()})
        pre_list.append({'index': index, 'realized_variance':pred_val.values.tolist()})


    # Create DataFrame with appropriate column names
    error_df = pd.DataFrame(error_list)
    realized_variance_df=pd.DataFrame(realized_variance_list)
    pre_list_df=pd.DataFrame(pre_list)

    return error_df,realized_variance_df,pre_list_df


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


def create_sequences(data, timesteps, prediction_length, shift):
    X, y = [], []

    for i in range(timesteps, len(data) - prediction_length, shift):
        # Extract the main data sequence
        main_sequence = np.array(data[(i - timesteps):i])

        # Debugging prints to check shapes and sample values
        print(f"main_sequence shape: {main_sequence.shape}")

        # Append the combined sequence to X
        X.append(main_sequence)

        # Define y as the target prediction sequence (first feature)
        y.append( np.array(data[i]).flatten() )  # Predicting the first feature

    X = np.array(X)
    y = np.array(y)

    return X, y

def preprocess_data(dataset_train, timesteps, prediction_length, shift):

    # Only select columns specified in the objective
    train_set = np.vstack(dataset_train['error'].values)

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

    # Number of features (indicators)
    indicators = 6

    return X_train, y_train, X_val, y_val, X_test, y_test, indicators, scaler

def preprocess_variance(dataset_train, timesteps, prediction_length, shift):

    # Only select columns specified in the objective
    train_set = np.vstack(dataset_train['realized_variance'].values)

    # Define sizes for validation, training, and test sets
    val_size = int(len(train_set) * 0.2)
    train_size = int(len(train_set) * 0.6)

    # Define validation, training, and test sets
    val_data = train_set[:val_size]
    train_data = train_set[val_size:val_size + train_size]
    test_data = train_set[val_size + train_size:]


    # Create sequences for each set
    X_train, y_train = create_sequences(train_data, timesteps, prediction_length, shift)
    X_val, y_val = create_sequences(val_data, timesteps, prediction_length, shift)
    X_test, y_test = create_sequences(test_data, timesteps, prediction_length, shift)

    # Number of features (indicators)
    indicators = 6

    return X_train, y_train, X_val, y_val, X_test, y_test, indicators


class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Define a trainable vector for attention
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1],), initializer='glorot_uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        # Compute attention scores
        score = K.tanh(K.dot(inputs, K.expand_dims(self.W)))
        attention_weights = K.softmax(score, axis=1)

        # Use broadcasting to align attention weights for multiplication with inputs
        attention_weights = K.repeat_elements(attention_weights, inputs.shape[-1], axis=-1)

        # Apply attention weights to inputs and compute context vector
        weighted_input = inputs * attention_weights
        context_vector = K.sum(weighted_input, axis=1)

        return context_vector



def build_GRU_model(X_train, y_train,indicators,timesteps,prediction_number,filters, kernel_size
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
    regressor.add(Dense(units=prediction_number))  # 预测一个值
    # 编译模型
    regressor.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='huber_loss')

    # loss='mean_squared_error'因为预测的是股票价格，是一个数值，类似于回归问题，就使用mse。（分类问题用cross——entropy）
    # 这里不用,metrics=['accuracy']，也是因为预测的是股票价格（而不是分类是否正确的问题）
    regressor.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    # Save the model if save_path is provided
    if True:
        model_path = os.path.join(r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\GRU_family_result', 'gru_error_model.h5')
        regressor.save(model_path)
        print(f"Model saved to {model_path}")

    return regressor


def make_predictions_separate(X_test, model, scaler, prediction_length=6):
    # Make predictions directly with preprocessed X_test
    predictions = model.predict(X_test)

    # Check if predictions match the expected shape
    if predictions.shape[1] != prediction_length:
        raise ValueError(f"Expected predictions to have {prediction_length} columns, but got {predictions.shape[1]}")

    # Initialize lists to store individual series for each prediction timestep
    predicted_series_list = [[] for _ in range(prediction_length)]

    # Separate predictions by timestep
    for i in range(len(predictions)):
        for j in range(prediction_length):
            predicted_series_list[j].append(predictions[i, j])

    # Prepare dummy arrays to match the scaler's expected input shape for inverse transformation
    predicted_series_inversed = []
    for i in range(prediction_length):
        # Create dummy array with the appropriate number of features
        dummy_predicted = np.zeros((len(predicted_series_list[i]), scaler.n_features_in_))

        # Fill the first column with the predicted values
        dummy_predicted[:, 0] = predicted_series_list[i]

        # Apply inverse transform and extract the first column
        predicted_values_inversed = scaler.inverse_transform(dummy_predicted)[:, 0]

        # Append inversed values to the list
        predicted_series_inversed.append(predicted_values_inversed)

    # Create DataFrame to store only the predicted values for each timestep
    dataframes = []
    for i in range(prediction_length):
        df = pd.DataFrame({
            f'Predicted_Value_Timestep_{i+1}': predicted_series_inversed[i]
        })
        dataframes.append(df)

    # Concatenate all DataFrames along columns
    final_dataframe = pd.concat(dataframes, axis=1)

    return final_dataframe



def plot_predictions_sep(final_dataframe, prediction_length=6, save_path='volatility_predictions.png',
                         scale_factor=1e5):
    """
    Plot the realized and predicted volatility values for each prediction timestep, with scaling and logarithmic y-axis.
    """
    plt.figure(figsize=(15, 10))

    # Loop through each prediction timestep
    for i in range(prediction_length):
        plt.subplot(prediction_length, 1, i + 1)

        # Extract and scale values
        real_values = final_dataframe[f'Real_Value_Timestep_{i + 1}'] * scale_factor
        predicted_values = final_dataframe[f'Predicted_Value_Timestep_{i + 1}'] * scale_factor

        # Plot realized and predicted volatility for the current timestep
        plt.plot(real_values, label=f'Realized Volatility (Timestep {i + 1})', color='blue')
        plt.plot(predicted_values, label=f'Predicted Volatility (Timestep {i + 1})', color='orange')

        # Add titles and labels
        plt.title(f'Volatility Prediction for Timestep {i + 1}')
        plt.xlabel('Index')
        plt.ylabel('Volatility (Scaled)')
        plt.yscale('log')  # Use logarithmic scale for y-axis
        plt.legend()
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(save_path)  # Save the plot image
    plt.show()

def create_final_dataframe(realized_variance_y_test, revised_variance_array, prediction_length=6):
    """
    Combines realized and predicted variance data into a single DataFrame for plotting.
    """
    # Initialize a dictionary to store columns for each timestep
    data = {}

    for i in range(prediction_length):
        # Add columns for each timestep's realized and predicted values
        data[f'Real_Value_Timestep_{i + 1}'] = realized_variance_y_test[:, i]
        data[f'Predicted_Value_Timestep_{i + 1}'] = revised_variance_array[:, i]

    # Convert to DataFrame
    final_dataframe = pd.DataFrame(data)

    return final_dataframe

def calibration_report_MSE_sep_with_save(realized_variance_y_test, revised_variance, prediction_length=6, save_path=None, scale_factor=1e5):
    """
    Calculates Mean Squared Error (MSE) between real and revised values for each prediction timestep,
    and saves the full MSE series as a DataFrame if a save path is provided.
    """
    # Verify that both arrays have the expected shape
    if realized_variance_y_test.shape != revised_variance.shape:
        raise ValueError("Shape mismatch: realized_variance_y_test and revised_variance must have the same shape.")

    mse_list = []
    mse_series = {}

    for i in range(prediction_length):
        real_values = realized_variance_y_test[:, i]
        revised_values = revised_variance[:, i]

        # Calculate Mean Squared Error (MSE)
        mse = np.mean((real_values - revised_values) ** 2)
        mse_list.append(mse)

        # Save the full error series for this timestep
        mse_series[f'Timestep_{i+1}'] = (real_values*scale_factor - revised_values*scale_factor) ** 2

        # Print error summary for each timestep
        print(f"Mean Squared Error (MSE) for Timestep {i+1}: {mse:.9f}")

    # Convert MSE series dictionary to a DataFrame
    mse_df = pd.DataFrame(mse_series)

    # Save the DataFrame as a CSV file if a save path is provided
    if True:
        mse_file_path = os.path.join(save_path, 'gru_av_error_mse_series.csv')
        mse_df.to_csv(mse_file_path, index=False)
        print(f"MSE series DataFrame saved to {mse_file_path}")

    return mse_list, mse_df


def calibration_report_MAPE_sep_with_save(realized_variance_y_test, revised_variance, prediction_length=6, save_path=None):
    """
    Calculates Mean Absolute Percentage Error (MAPE) between real and revised values for each prediction timestep,
    and saves the full MAPE series as a DataFrame if a save path is provided.
    """
    # Verify that both arrays have the expected shape
    if realized_variance_y_test.shape != revised_variance.shape:
        raise ValueError("Shape mismatch: realized_variance_y_test and revised_variance must have the same shape.")

    mape_list = []
    mape_series = {}

    for i in range(prediction_length):
        real_values = realized_variance_y_test[:, i]
        revised_values = revised_variance[:, i]

        # Calculate Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((real_values - revised_values) / real_values))
        mape_list.append(mape)

        # Save the full error series for this timestep
        mape_series[f'Timestep_{i+1}'] = np.abs((real_values - revised_values) / real_values)

        # Print error summary for each timestep
        print(f"Mean Absolute Percentage Error (MAPE) for Timestep {i+1}: {mape:.9f}%")

    # Convert MAPE series dictionary to a DataFrame
    mape_df = pd.DataFrame(mape_series)

    # Save the DataFrame as a CSV file if a save path is provided
    if save_path:
        mape_file_path = os.path.join(save_path, 'gru_error_mape_series.csv')
        mape_df.to_csv(mape_file_path, index=False)
        print(f"MAPE series DataFrame saved to {mape_file_path}")

    return mape_list, mape_df

if __name__ == "__main__":
    portfolio = pd.read_csv(r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\portfolio\Portfolio_agg.csv')
    result = pd.read_csv(
        r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\Forecast_result\avgarch_results.csv')
    figarch_forecasts = pd.read_csv(
        r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\Forecast_result\avGARCH_forecasts.csv')

    expanded_result, volatility, residuals = expand_volatility_residuals(result)
    error, realized_variance_df,pre_list_df= extract_error(figarch_forecasts, portfolio)

    timesteps = 30  # 单次训练步长
    shift= 1  #rolling shift
    prediction_length = 1 #单次预测步长
    prediction_number=6
    datasets = error.copy()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 若设置为‘-1’则是使用CPU，忽略GPU
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices(
        'GPU')))  # 如果输出是 Num GPUs Available: 0，这意味着 TensorFlow 将只使用 CPU 运行。

    (X_train, y_train, X_val, y_val, X_test, y_test, indicators,sc1) = preprocess_data(datasets,
                                                            timesteps, prediction_length, shift)

    (realized_variance_x_train, realized_variance_y_train, realized_variance_x_val,
     realized_variance_y_val, realized_variance_x_test, realized_variance_y_test, indicators) = (
        preprocess_variance(realized_variance_df,timesteps, prediction_length, shift))

    (pre_variance_x_train, pre_variance_y_train, pre_variance_x_val,
     pre_variance_y_val, pre_variance_x_test,pre_variance_y_test, indicators) = (
        preprocess_variance(pre_list_df, timesteps, prediction_length, shift))



    regressor = build_GRU_model(X_train, y_train,indicators, timesteps, prediction_number,filters=64, kernel_size=3,
                                 units=256, dropout_rate=0.2, epochs=100, batch_size= 32)
    #Best parameters: {'activation': 'tanh', 'batch_size': 16, 'dropout_rate': 0.4, 'epochs': 50, 'layer_numbers': 3,
       #          'units': 128}

    '''
    # 得到predicted_error 绘制对比图--------only need the first value of prediction,since shift=1,predict_length=6
    real_price, predicted_price = make_predictions_all (X_test,y_test,regressor,sc)
    plot_predictions(real_price, predicted_price)
    print(calibration_report_MAPE(real_price, predicted_price))
    print(calibration_report_MSE(real_price, predicted_price))
    '''

    predicted_error=make_predictions_separate(X_test, regressor,sc1,prediction_number)

    revised_variance=pre_variance_y_test+ predicted_error

    # Convert revised_variance DataFrame to numpy array
    revised_variance_array = revised_variance.values

    save_path = r"C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\error_series"
    # Calculate MSE and MAPE
    mse_list, mse_df = calibration_report_MSE_sep_with_save(realized_variance_y_test, revised_variance_array,
                                                            prediction_length=6, save_path=save_path)
    mape_list, mape_df = calibration_report_MAPE_sep_with_save(realized_variance_y_test, revised_variance_array,
                                                               prediction_length=6, save_path=save_path)
    print(mape_list)
    print(mse_list)


    # Generate final_dataframe for plotting
    final_dataframe = create_final_dataframe(realized_variance_y_test, revised_variance_array, prediction_length=6)

    # Plot predictions separately
    plot_predictions_sep(final_dataframe, prediction_length=6,
            save_path=r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\GRU_family_result\gru_av_error_predictions.png')









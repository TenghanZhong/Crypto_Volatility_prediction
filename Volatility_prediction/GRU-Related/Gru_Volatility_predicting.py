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
import json

def create_sequences(data, timesteps, prediction_length, shift):
    X, y = [], []
    for i in range(timesteps, len(data) - prediction_length, shift):
        X.append(data[(i - timesteps):i, :])
        y.append(data[i:i + prediction_length, 0])  # Predicting the first feature
    X = np.array(X)
    y = np.array(y)
    return X, y


def preprocess_data(dataset_train, timesteps, prediction_length, shift, objective):
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



def build_GRU_model(X_train, y_train,indicators,timesteps,prediction_length,filters, kernel_size
    , units, dropout_rate,epochs,batch_size):

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 若设置为‘-1’则是使用CPU，忽略GPU
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices(
        'GPU')))  # 如果输出是 Num GPUs Available: 0，这意味着 TensorFlow 将只使用 CPU 运行。

    regressor = Sequential()


    input_shape = (timesteps, indicators)
    #X_train.shape[1]=timesteps
    # 第一层 GRU
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


def calibration_report_MSE_sep(final_dataframe, prediction_length=6):
    """
    Calculates Mean Squared Error (MSE) between real and predicted values for each prediction timestep.
    """
    mse_list = []
    for i in range(prediction_length):
        real_price = final_dataframe[f'Real_Value_Timestep_{i+1}'].values.flatten()
        predicted_price = final_dataframe[f'Predicted_Value_Timestep_{i+1}'].values.flatten()

        if len(real_price) != len(predicted_price):
            raise ValueError("The lengths of real_price and predicted_price must be the same")

        # Calculate Mean Squared Error (MSE)
        mse = np.mean((real_price - predicted_price) ** 2)
        mse_list.append(mse)

        # Print error summary for each timestep
        print(f"Mean Squared Error (MSE) for Timestep {i+1}: {mse:.9f}")
    return mse_list

def calibration_report_MAPE_sep(final_dataframe, prediction_length=6):
    """
    Calculates Mean Absolute Percentage Error (MAPE) between real and predicted values for each prediction timestep.
    """
    mape_list = []
    for i in range(prediction_length):
        real_price = final_dataframe[f'Real_Value_Timestep_{i+1}'].values.flatten()
        predicted_price = final_dataframe[f'Predicted_Value_Timestep_{i+1}'].values.flatten()

        if len(real_price) != len(predicted_price):
            raise ValueError("The lengths of real_price and predicted_price must be the same")

        # Calculate Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((real_price - predicted_price) / real_price)) * 100
        mape_list.append(mape)

        # Print error summary for each timestep
        print(f"Mean Absolute Percentage Error (MAPE) for Timestep {i+1}: {mape:.9f}%")
    return mape_list




def plot_predictions(real_price, predicted_price):
    """
    Plots real vs. predicted values for volatility.
    """
    plt.figure(figsize=(18, 8))
    plt.plot(real_price, color='red', label='Real')
    plt.plot(predicted_price, color='blue', label='Predicted')
    plt.title('Volatility Prediction')
    plt.xlabel('Time')
    plt.ylabel('Volatility')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

def calibration_report_MSE(real_price, predicted_price):
    """
    Calculates Mean Squared Error (MSE) between real and predicted values.
    """
    real_price = np.array(real_price).flatten()
    predicted_price = np.array(predicted_price).flatten()

    if len(real_price) != len(predicted_price):
        raise ValueError("The lengths of real_price and predicted_price must be the same")

    # Calculate Mean Squared Error (MSE)
    mse = np.mean((real_price - predicted_price) ** 2)

    # Print and return error summary
    error_summary = f"Mean Squared Error (MSE) of real and predicted: {mse:.9f}"
    print(error_summary)
    return error_summary

def calibration_report_MAPE(real_price, predicted_price):
    """
    Calculates Mean Absolute Percentage Error (MAPE) between real and predicted values.
    """
    real_price = np.array(real_price).flatten()
    predicted_price = np.array(predicted_price).flatten()

    if len(real_price) != len(predicted_price):
        raise ValueError("The lengths of real_price and predicted_price must be the same")

    # Calculate Mean Absolute Percentage Error (MAPE)
    mape = np.mean(np.abs((real_price - predicted_price) / real_price)) * 100

    # Print and return error summary
    error_summary = f"Mean Absolute Percentage Error (MAPE) of real and predicted: {mape:.9f}%"
    print(error_summary)
    return error_summary


if __name__ == "__main__":
    portfolio = pd.read_csv(r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\portfolio\Portfolio_agg.csv')

    # 分阶段优化
    objective = ('Realized_Variance')  # 要预测的值为第一个
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

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 若设置为‘-1’则是使用CPU，忽略GPU
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices(
        'GPU')))  # 如果输出是 Num GPUs Available: 0，这意味着 TensorFlow 将只使用 CPU 运行。


    X_train, y_train, X_val, y_val, X_test, y_test, indicators ,sc = preprocess_data(datasets,
                                                            timesteps, prediction_length, shift, objective)
    # 训练模型,分别进行增量学习
    regressor = build_GRU_model(X_train, y_train,indicators, timesteps, prediction_length,filters=64, kernel_size=3,
                                 units=256, dropout_rate=0.2, epochs=50, batch_size= 16)

    #parameters: {'activation': 'tanh', 'batch_size': 16, 'dropout_rate': 0.4, 'epochs': 50, 'layer_numbers': 2,
                # 'units': 256}
    #Best MAE: 0.33480495010882055
    '''
    # 得到predicted_error 绘制对比图--------only need the first value of prediction,since shift=1,predict_length=6
    real_price, predicted_price = make_predictions_all (X_test,y_test,regressor,sc)
    plot_predictions(real_price, predicted_price)
    print(calibration_report_MAPE(real_price, predicted_price))
    print(calibration_report_MSE(real_price, predicted_price))
    '''

    data_comparision=make_predictions_separate(X_test, y_test,regressor,sc, prediction_length)

    plot_predictions_sep(data_comparision, prediction_length)
    mape_list=calibration_report_MAPE_sep(data_comparision, prediction_length)
    print(mape_list)
    mse_list=calibration_report_MSE_sep(data_comparision, prediction_length)
    print(mse_list)


    plot_predictions_sep(data_comparision, prediction_length,
                         save_path=r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\GRU_family_result\gru_volatility_predictions.png')

    # 定义保存路径
    save_dir = r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\GRU_family_result'

    # 检查保存路径是否存在，不存在则创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 保存训练好的模型
    model_path = os.path.join(save_dir, 'gru_model.h5')
    regressor.save(model_path)  # 将模型保存到指定路径

    # 将 MAPE 和 MSE 列表保存为 JSON 文件
    mape_path = os.path.join(save_dir, 'gru_mape_list.json')
    with open(mape_path, 'w') as f:
        json.dump(mape_list, f)

    mse_path = os.path.join(save_dir, 'gru_mse_list.json')
    with open(mse_path, 'w') as f:
        json.dump(mse_list, f)







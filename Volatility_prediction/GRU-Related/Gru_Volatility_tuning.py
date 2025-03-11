import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, GRU, Dropout, Conv1D, MaxPooling1D, Layer
from keras import backend as K
from keras.callbacks import EarlyStopping
from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV
import os
import tensorflow as tf
import json
from datetime import datetime

from sklearn.preprocessing import StandardScaler

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
    train_set = dataset_train[list(objective)].values

    # Number of features (indicators)
    indicators = train_set.shape[1]

    # Split the dataset into training, validation, and test sets first
    train_size = int(len(train_set) * 0.6)
    val_size = int(len(train_set) * 0.2)

    # Define train, validation, and test sets by directly slicing the array
    train_data = train_set[:train_size]
    val_data = train_set[train_size:train_size + val_size]
    test_data = train_set[train_size + val_size:]

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
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1],), initializer='glorot_uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        score = K.tanh(K.dot(inputs, K.expand_dims(self.W)))
        attention_weights = K.softmax(score, axis=1)
        attention_weights = K.repeat_elements(attention_weights, inputs.shape[-1], axis=-1)
        weighted_input = inputs * attention_weights
        context_vector = K.sum(weighted_input, axis=1)
        return context_vector

def build_regressor(optimizer='adam', units=256, dropout_rate=0.2, layer_numbers=3,
                    activation='tanh', prediction_length=6, filters=64, kernel_size=3):
    regressor = Sequential()

    # Dynamically add GRU layers
    for _ in range(layer_numbers):
        regressor.add(GRU(units=units, activation=activation, return_sequences=True))
        regressor.add(Dropout(rate=dropout_rate))

    # Final GRU layer without return_sequences
    regressor.add(GRU(units=units, activation=activation, return_sequences=False))
    regressor.add(Dropout(rate=dropout_rate))

    # Add the Attention Layer
    #regressor.add(AttentionLayer())

    regressor.add(Dense(units=prediction_length))  # Output layer
    regressor.compile(optimizer=optimizer, loss='huber_loss')  # mean_absolute_error can also be used
    return regressor

if __name__ == "__main__":
    portfolio = pd.read_csv(r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\portfolio\Portfolio_agg.csv')

    objective = ('Realized_Variance', 'Volume')
    timesteps, shift, prediction_length = 120, 1, 6
    datasets = portfolio.copy()
    datasets.replace([np.inf, -np.inf], np.nan, inplace=True)
    datasets.interpolate(method='linear', inplace=True)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    X_train, y_train, X_val, y_val, X_test, y_test, indicators,sc = preprocess_data(datasets,
                                                            timesteps, prediction_length, shift, objective)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # Revised KerasRegressor instantiation
    regressor = KerasRegressor(
        model=build_regressor,
        units=256,
        dropout_rate=0.2,
        layer_numbers=3,
        activation='tanh',
        optimizer='adam',
        epochs=50,
        batch_size=16,
        filters=64,  # Set filters here
        kernel_size=3  # Set kernel_size here
    )

    # Adjust parameters dictionary for GridSearchCV
    parameters_dictionary = {
        'batch_size': [16, 32],
        'epochs': [50, 25],
        'units': [256, 128],
        'dropout_rate': [0.2, 0.4],
        'layer_numbers': [2,3],
        'activation': ['tanh']
    }

    # Proceed with GridSearchCV
    grid_search = GridSearchCV(estimator=regressor, param_grid=parameters_dictionary, scoring='neg_mean_absolute_error',
                               cv=5)

    #Best parameters: {'activation': 'tanh', 'batch_size': 32, 'dropout_rate': 0.4, 'epochs': 50, 'layer_numbers': 1, 'units': 256}


    grid_search.fit(X_val, y_val)

    # Get the best parameters and score
    best_parameters = grid_search.best_params_
    best_accuracy = grid_search.best_score_

    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save results to a JSON file with a timestamp in the filename
    results = {
        "Best Parameters": best_parameters,
        "Best MAE": -best_accuracy
    }

    filename = f"grid_search_results_{timestamp}.json"
    with open(filename, 'w') as file:
        json.dump(results, file, indent=4)

    # Print results
    print(f"Best parameters: {best_parameters}")
    print(f"Best MAE: {-best_accuracy}")
    print(f"Results saved to {filename}")


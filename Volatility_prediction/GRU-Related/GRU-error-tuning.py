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

def extract_error(figarch_forecasts, portfolio):
    error_list = []

    for i in range(len(figarch_forecasts) - 6):
        index = figarch_forecasts['index'][i]
        pred_val = figarch_forecasts.iloc[i, 1:]
        realized_variance = portfolio.iloc[index + 1:index + 7]['Realized_Variance']

        # Calculate the error and ensure the length matches
        if len(pred_val) == len(realized_variance):
            error = (pred_val.values - realized_variance.values).tolist()
            error_list.append({'index': index, 'error': error})

    # Create DataFrame with appropriate column names
    error_df = pd.DataFrame(error_list)

    return error_df


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

        # Append the combined sequence to X
        X.append(main_sequence)

        # Define y as the target prediction sequence (first feature)
        y.append( np.array(data[i]).flatten() )  # Predicting the first feature

    X = np.array(X)
    y = np.array(y)

    return X, y

def preprocess_data(dataset_train, timesteps, prediction_length, shift, objective):

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

    return X_train, y_train, X_val, y_val, X_test, y_test, indicators

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
                    activation='tanh', prediction_number=6, filters=64, kernel_size=3, timesteps=30, features=6):
    regressor = Sequential()

    # Add GRU layers as specified, setting input_shape in the first layer
    regressor.add(GRU(units=units, activation=activation, return_sequences=True, input_shape=(timesteps, features)))
    regressor.add(Dropout(rate=dropout_rate))

    for _ in range(layer_numbers - 1):
        regressor.add(GRU(units=units, activation=activation, return_sequences=True))
        regressor.add(Dropout(rate=dropout_rate))

    # Final GRU layer without return_sequences
    regressor.add(GRU(units=units, activation=activation, return_sequences=False))
    regressor.add(Dropout(rate=dropout_rate))

    # Output layer with units equal to the number of features (6)
    regressor.add(Dense(units=prediction_number))  # Predicts 6 values as a flat output
    regressor.compile(optimizer=optimizer, loss='huber_loss')

    return regressor


if __name__ == "__main__":
    portfolio = pd.read_csv(r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\portfolio\Portfolio_agg.csv')
    result = pd.read_csv(
        r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\Forecast_result\figarch_results.csv')
    figarch_forecasts= pd.read_csv(r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\Forecast_result\FIGARCH_forecasts.csv')

    expanded_result,volatility, residuals=expand_volatility_residuals(result)
    error=extract_error(figarch_forecasts, portfolio)

    objective = ('error')
    timesteps, shift, prediction_length,prediction_number = 30, 1, 1 ,6
    datasets = error.copy()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    (X_train, y_train, X_val, y_val, X_test, y_test, indicators) = preprocess_data(datasets,timesteps, prediction_length, shift, objective)

    # Debugging checks for X_val and y_val
    print(f"X_val shape: {X_val.shape}")  # Expecting (num_samples, timesteps, features)
    print(f"y_val shape: {y_val.shape}")  # Expecting (num_samples, prediction_length)

    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    # Revised KerasRegressor instantiation
    # Instantiate the KerasRegressor with verbose=0 to reduce log clutter
    regressor = KerasRegressor(
        model=build_regressor,
        units=256,
        dropout_rate=0.2,
        layer_numbers=3,
        activation='tanh',
        optimizer='adam',
        epochs=50,
        batch_size=16,
        filters=64,
        kernel_size=3,
        timesteps=30,  # Specify timesteps explicitly
        features=6,  # Specify features explicitly
        prediction_number=6  # Ensure output matches target shape
    )

    # Adjust parameters dictionary for GridSearchCV
    parameters_dictionary = {
        'batch_size': [16, 32],
        'epochs': [50, 75],
        'units': [256],
        'dropout_rate': [0.2,0.3, 0.4],
        'layer_numbers': [3,4],
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


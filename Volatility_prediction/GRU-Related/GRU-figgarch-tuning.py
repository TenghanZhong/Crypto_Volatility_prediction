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


def create_sequences(data, timesteps, prediction_length, shift, volatility, residuals):
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

def preprocess_data(dataset_train, timesteps, prediction_length, shift, objective, volatility, residuals):
    # Convert 'Hour' column to a datetime format and sort by it
    dataset_train['Hour'] = pd.to_datetime(dataset_train['Hour'].astype(str)).dt.strftime('%Y%m%d')
    dataset_train = dataset_train.sort_values(by='Hour', ascending=True).reset_index(drop=True)

    # Only select columns specified in the objective
    train_set = dataset_train[list(objective)].values

    # Split the dataset into training, validation, and test sets
    train_size = int(len(train_set) * 0.6)
    val_size = int(len(train_set) * 0.2)

    # Define train, validation, and test sets
    train_data = train_set[:train_size]
    val_data = train_set[train_size:train_size + val_size]
    test_data = train_set[train_size + val_size:]

    # Scale the main data based on training data only
    scaler_main = StandardScaler()
    train_data_scaled = scaler_main.fit_transform(train_data)
    val_data_scaled = scaler_main.transform(val_data)
    test_data_scaled = scaler_main.transform(test_data)

    # Scale the volatility and residuals independently based on their respective training parts
    scaler_volatility = StandardScaler()
    scaler_residuals = StandardScaler()
    volatility_train_scaled = scaler_volatility.fit_transform(volatility[:train_size])
    residuals_train_scaled = scaler_residuals.fit_transform(residuals[:train_size])

    # Scale validation and test sets for volatility and residuals
    volatility_val_scaled = scaler_volatility.transform(volatility[train_size:train_size + val_size])
    volatility_test_scaled = scaler_volatility.transform(volatility[train_size + val_size:])

    residuals_val_scaled = scaler_residuals.transform(residuals[train_size:train_size + val_size])
    residuals_test_scaled = scaler_residuals.transform(residuals[train_size + val_size:])

    # Create sequences for each set
    X_train, y_train = create_sequences(train_data_scaled, timesteps, prediction_length, shift, volatility_train_scaled, residuals_train_scaled)
    X_val, y_val = create_sequences(val_data_scaled, timesteps, prediction_length, shift, volatility_val_scaled, residuals_val_scaled)
    X_test, y_test = create_sequences(test_data_scaled, timesteps, prediction_length, shift, volatility_test_scaled, residuals_test_scaled)

    # Number of features (indicators)
    indicators = X_train.shape[2]

    return X_train, y_train, X_val, y_val, X_test, y_test, indicators, scaler_main, scaler_volatility, scaler_residuals

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

    '''
    # CNN layer: 1D Convolution followed by MaxPooling
    regressor.add(Conv1D(filters=filters, kernel_size=kernel_size, activation='relu', input_shape=(timesteps, indicators)))
    regressor.add(MaxPooling1D(pool_size=2))
    regressor.add(Dropout(rate=dropout_rate))
    '''

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
    result = pd.read_csv(
        r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\Forecast_result\figarch_results.csv')

    expanded_result,volatility, residuals=expand_volatility_residuals(result)


    objective = ('Realized_Variance', 'Volume')
    timesteps, shift, prediction_length = 120, 1, 6
    datasets = portfolio.copy()
    datasets.replace([np.inf, -np.inf], np.nan, inplace=True)
    datasets.interpolate(method='linear', inplace=True)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    (X_train, y_train, X_val, y_val, X_test, y_test, indicators,sc,
     scaler_volatility, scaler_residuals) = preprocess_data(datasets,
                     timesteps, prediction_length, shift, objective, volatility, residuals)

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


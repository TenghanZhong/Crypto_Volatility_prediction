import pandas as pd
import openpyxl
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

# Load the data
file_path = r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\portfolio\portfolio_agg.csv'
df = pd.read_csv(file_path)

# Load the FIGARCH parameters file
figarch_params_file = r'C:\Users\26876\Desktop\lstmgarch_project\Original_crydata\Forecast_result\figarch_results.csv'
figarch_params_df = pd.read_csv(figarch_params_file)

# Prepare the data
returns = df['Returns'].values
realized_variance = df['Realized_Variance'].values

# Split the data into training, validation, and test sets
train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

train_size = int(len(returns) * train_ratio)
val_size = int(len(returns) * val_ratio)

train_returns = returns[:train_size]
val_returns = returns[train_size:train_size + val_size]
test_returns = returns[train_size + val_size:]

train_realized_variance = realized_variance[:train_size]
val_realized_variance = realized_variance[train_size:train_size + val_size]
test_realized_variance = realized_variance[train_size + val_size:]


# Define the Enhanced SPINN model
class EnhancedSPINN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers=3):
        super(EnhancedSPINN, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]

        # Add more hidden layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# Define the FIGARCH loss function
# This loss function includes a FIGARCH-like term to enforce variance dynamics
def figarch_loss(predicted_variance, returns, alpha0, alpha1, d, beta, epsilon=1e-6):
    # Add epsilon to avoid division by zero or taking negative values to fractional powers
    predicted_variance = torch.clamp(predicted_variance, min=epsilon)
    figarch_term = alpha0 + alpha1 * (returns ** 2) + beta * (predicted_variance ** d)
    loss = torch.mean((predicted_variance - figarch_term) ** 2)
    return loss


# Set up the model, loss, and optimizer
input_size = 5  # 1 for returns, 4 for FIGARCH parameters
hidden_size = 20
output_size = 1
num_hidden_layers = 4

model = EnhancedSPINN(input_size, hidden_size, output_size, num_hidden_layers)
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)  # Add L2 regularization

# Training loop with rolling window
epochs = 500
window_size = 120
prediction_horizon = 6

# DataFrame to store predictions and HAPE
predictions_list = []
hape_list = []

# Best loss tracking
best_loss = float('inf')
best_model_state = None

# Rolling window training for training set
for start in range(0, len(train_returns) - window_size - prediction_horizon + 1):
    end = start + window_size

    # Extract the current window of returns and realized variance for training
    x_window = train_returns[start:end]
    y_window = train_realized_variance[end:end + prediction_horizon]

    # Compute mean and standard deviation for the current window
    window_returns_mean = x_window.mean()
    window_returns_std = x_window.std()
    window_variance_mean = y_window.mean()
    window_variance_std = y_window.std()

    # Normalize the data for the current window
    x_window_normalized = (x_window - window_returns_mean) / window_returns_std
    y_window_normalized = (y_window - window_variance_mean) / window_variance_std

    # Convert to tensors
    x_window_tensor = torch.tensor(x_window_normalized, dtype=torch.float32).view(-1, 1)
    y_window_tensor = torch.tensor(y_window_normalized, dtype=torch.float32).view(-1, 1)

    # Extract FIGARCH parameters for the current window directly from the DataFrame
    alpha0 = figarch_params_df.loc[start, 'omega'] if 'omega' in figarch_params_df.columns else 0.0001
    alpha1 = figarch_params_df.loc[start, 'phi'] if 'phi' in figarch_params_df.columns else 0.85
    d = figarch_params_df.loc[
        start, 'd'] if 'd' in figarch_params_df.columns else 0.4  # Default to 0.4 if 'd' not found
    beta = figarch_params_df.loc[
        start, 'beta'] if 'beta' in figarch_params_df.columns else 0.05  # Default to 0.05 if 'beta' not found

    figarch_params_window = torch.tensor([alpha0, alpha1, d, beta], dtype=torch.float32).view(1, -1).repeat(
        x_window_tensor.size(0), 1)

    for epoch in range(epochs):
        model.train()

        # Convert returns to PyTorch Variable
        returns_var = Variable(x_window_tensor, requires_grad=True)

        # Concatenate returns with FIGARCH parameters as inputs
        inputs = torch.cat((returns_var, figarch_params_window), dim=1)

        # Forward pass
        predicted_variance = model(inputs)

        # Inverse transform the variance prediction
        predicted_variance_denormalized = predicted_variance[
                                          -prediction_horizon:] * window_variance_std + window_variance_mean
        y_window_denormalized = y_window_tensor * window_variance_std + window_variance_mean

        # Calculate HAPE for the predicted values (add epsilon to avoid division by zero)
        hape = torch.abs(predicted_variance_denormalized - y_window_denormalized) / (y_window_denormalized + 1e-6)
        data_loss = torch.mean(hape)

        # Physics-based loss using FIGARCH dynamics
        physics_loss = figarch_loss(predicted_variance[-prediction_horizon:], returns_var[-prediction_horizon:], alpha0,
                                    alpha1, d, beta)

        # Total loss as a weighted sum of data loss and physics loss
        total_loss = data_loss + physics_loss

        # Check for NaN loss and break if encountered
        if torch.isnan(total_loss):
            print(f'NaN loss encountered at window [{start}-{end}], epoch [{epoch + 1}]')
            break

        # Backward pass and optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Track the best loss and save the model state
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_model_state = copy.deepcopy(model.state_dict())

        # Print the loss every 50 epochs
        if (epoch + 1) % 50 == 0:
            print(
                f'Window [{start}-{end}], Epoch [{epoch + 1}/{epochs}], Data Loss: {data_loss.item():.4f}, Physics Loss: {physics_loss.item():.4f}, Total Loss: {total_loss.item():.4f}')

    # Store the predicted variance and corresponding HAPE for the 6-hour prediction horizon
    predictions_list.append(predicted_variance_denormalized.detach().numpy().flatten())
    hape_list.append(hape.detach().numpy().flatten())

# Load the best model state
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# Evaluate on validation set
val_predictions_list = []
val_hape_list = []
for start in range(0, len(val_returns) - window_size - prediction_horizon + 1):
    end = start + window_size

    # Extract the current window of returns and realized variance for validation
    x_window = val_returns[start:end]
    y_window = val_realized_variance[end:end + prediction_horizon]

    # Normalize the data for the current window
    window_returns_mean = x_window.mean()
    window_returns_std = x_window.std()
    window_variance_mean = y_window.mean()
    window_variance_std = y_window.std()

    x_window_normalized = (x_window - window_returns_mean) / window_returns_std
    y_window_normalized = (y_window - window_variance_mean) / window_variance_std

    # Convert to tensors
    x_window_tensor = torch.tensor(x_window_normalized, dtype=torch.float32).view(-1, 1)
    y_window_tensor = torch.tensor(y_window_normalized, dtype=torch.float32).view(-1, 1)

    # Extract FIGARCH parameters for the current window directly from the DataFrame
    alpha0 = figarch_params_df.loc[start, 'omega'] if 'omega' in figarch_params_df.columns else 0.0001
    alpha1 = figarch_params_df.loc[start, 'phi'] if 'phi' in figarch_params_df.columns else 0.85
    d = figarch_params_df.loc[start, 'd'] if 'd' in figarch_params_df.columns else 0.4
    beta = figarch_params_df.loc[start, 'beta'] if 'beta' in figarch_params_df.columns else 0.05

    figarch_params_window = torch.tensor([alpha0, alpha1, d, beta], dtype=torch.float32).view(1, -1).repeat(
        x_window_tensor.size(0), 1)

    # Concatenate returns with FIGARCH parameters as inputs
    inputs = torch.cat((x_window_tensor, figarch_params_window), dim=1)

    # Forward pass
    model.eval()
    with torch.no_grad():
        predicted_variance = model(inputs)

    # Denormalize the predictions
    predicted_variance_denormalized = predicted_variance[
                                      -prediction_horizon:] * window_variance_std + window_variance_mean
    future_y_window = val_realized_variance[end:end + prediction_horizon]
    future_y_window_tensor = torch.tensor(future_y_window, dtype=torch.float32).view(-1, 1)
    future_y_window_denormalized = future_y_window_tensor * window_variance_std + window_variance_mean

    # Calculate HAPE for validation
    hape_future = torch.abs(predicted_variance_denormalized - future_y_window_denormalized) / (
                future_y_window_denormalized + 1e-6)
    val_predictions_list.append(predicted_variance_denormalized.detach().numpy().flatten())
    val_hape_list.append(hape_future.detach().numpy().flatten())

# Save validation predictions and HAPE into an Excel file
with pd.ExcelWriter('validation_predicted_variance_and_hape3.xlsx') as writer:
    pd.DataFrame(val_predictions_list).to_excel(writer, sheet_name='Validation_Predicted_Variance', index=False)
    pd.DataFrame(val_hape_list).to_excel(writer, sheet_name='Validation_HAPE', index=False)

# Evaluate on test set
test_predictions_list = []
test_hape_list = []
for start in range(0, len(test_returns) - window_size - prediction_horizon + 1):
    end = start + window_size

    # Extract the current window of returns and realized variance for testing
    x_window = test_returns[start:end]
    y_window = test_realized_variance[end:end + prediction_horizon]

    # Normalize the data for the current window
    window_returns_mean = x_window.mean()
    window_returns_std = x_window.std()
    window_variance_mean = y_window.mean()
    window_variance_std = y_window.std()

    x_window_normalized = (x_window - window_returns_mean) / window_returns_std
    y_window_normalized = (y_window - window_variance_mean) / window_variance_std

    # Convert to tensors
    x_window_tensor = torch.tensor(x_window_normalized, dtype=torch.float32).view(-1, 1)
    y_window_tensor = torch.tensor(y_window_normalized, dtype=torch.float32).view(-1, 1)

    # Extract FIGARCH parameters for the current window directly from the DataFrame
    alpha0 = figarch_params_df.loc[start, 'omega'] if 'omega' in figarch_params_df.columns else 0.0001
    alpha1 = figarch_params_df.loc[start, 'phi'] if 'phi' in figarch_params_df.columns else 0.85
    d = figarch_params_df.loc[start, 'd'] if 'd' in figarch_params_df.columns else 0.4
    beta = figarch_params_df.loc[start, 'beta'] if 'beta' in figarch_params_df.columns else 0.05

    figarch_params_window = torch.tensor([alpha0, alpha1, d, beta], dtype=torch.float32).view(1, -1).repeat(
        x_window_tensor.size(0), 1)

    # Concatenate returns with FIGARCH parameters as inputs
    inputs = torch.cat((x_window_tensor, figarch_params_window), dim=1)

    # Forward pass
    model.eval()
    with torch.no_grad():
        predicted_variance = model(inputs)

    # Denormalize the predictions
    predicted_variance_denormalized = predicted_variance[
                                      -prediction_horizon:] * window_variance_std + window_variance_mean
    future_y_window = test_realized_variance[end:end + prediction_horizon]
    future_y_window_tensor = torch.tensor(future_y_window, dtype=torch.float32).view(-1, 1)
    future_y_window_denormalized = future_y_window_tensor * window_variance_std + window_variance_mean

    # Calculate HAPE for testing
    hape_future = torch.abs(predicted_variance_denormalized - future_y_window_denormalized) / (
                future_y_window_denormalized + 1e-6)
    test_predictions_list.append(predicted_variance_denormalized.detach().numpy().flatten())
    test_hape_list.append(hape_future.detach().numpy().flatten())[1,2,3,4,5,6,1,2,3,4,5,6]

# Save test predictions and HAPE into an Excel file
with pd.ExcelWriter('test_predicted_variance_and_hape4.xlsx') as writer:
    pd.DataFrame(test_predictions_list).to_excel(writer, sheet_name='Test_Predicted_Variance', index=False)
    pd.DataFrame(test_hape_list).to_excel(writer, sheet_name='Test_HAPE', index=False)

with pd.ExcelWriter('validation_predicted_variance_and_hape4.xlsx') as writer:
    pd.DataFrame(val_predictions_list).to_excel(writer, sheet_name='Validation_Predicted_Variance', index=False)
    pd.DataFrame(val_hape_list).to_excel(writer, sheet_name='Validation_HAPE', index=False)

# Save predictions and HAPE into an Excel file
with pd.ExcelWriter('predicted_variance_and_hape4.xlsx') as writer:
    pd.DataFrame(predictions_list).to_excel(writer, sheet_name='Predicted_Variance', index=False)
    pd.DataFrame(hape_list).to_excel(writer, sheet_name='HAPE', index=False)

# Convert to numpy and display the first few predictions
predicted_variance_np = predicted_variance_denormalized.detach().numpy()
print(predicted_variance_np[:5])
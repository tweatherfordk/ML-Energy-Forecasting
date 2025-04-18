import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Flatten, Dense, LSTM, Dropout, Concatenate, Reshape
import numpy as np

# Load npz data file
data = np.load('preprocessed_energy_data.npz')

# Manually map the arrays to their correct variable names
train_X = data['X_train']
train_y = data['y_train']
val_X = data['X_val']
val_y = data['y_val']
test_X = data['X_test']
test_y = data['y_test']

print(train_X.shape)
# Reshape data
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))  # (samples, timesteps, features)
val_X = val_X.reshape((val_X.shape[0], 1, val_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

# Verify that the data is loaded correctly
print(f"train_X shape: {train_X.shape}, train_y shape: {train_y.shape}")
print(f"val_X shape: {val_X.shape}, val_y shape: {val_y.shape}")
print(f"test_X shape: {test_X.shape}, test_y shape: {test_y.shape}")

def build_cnn_lstm_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # First CNN Block
    x = Conv1D(filters=48, kernel_size=3, activation='relu', padding='same')(inputs)
    x1 = Flatten()(x)
    x1 = Dense(24, activation='relu')(x1)
    
    # Second CNN Block
    x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
    x2 = Flatten()(x)
    x2 = Dense(24, activation='relu')(x2)
    
    # Third CNN Block
    x = Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(x)
    x3 = Flatten()(x)
    x3 = Dense(24, activation='relu')(x3)
    
    # Dropout Layer
    x = Dropout(0.25)(x3)
    
    # Reshape layer to match LSTM input (timesteps, features)
    x = Reshape((-1, 24))(x)  # Reshape to 3D: (batch_size, timesteps, features)
    
    # LSTM Layers
    x = LSTM(20, return_sequences=True)(x)
    x = LSTM(20, return_sequences=True)(x)
    x = LSTM(20, return_sequences=False)(x)
    
    # Dropout Layer
    x = Dropout(0.25)(x)
    
    # Concatenate CNN and LSTM Outputs
    x = Concatenate()([x, x1, x2, x3])
    
    # Final Dense Layer
    output = Dense(1, activation='tanh')(x)
    
    model = Model(inputs=inputs, outputs=output)
    return model

# Define model
input_shape = train_X.shape[1:]  # Should be (1, 12)
print(input_shape)
model = build_cnn_lstm_model(input_shape)

# Compile model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train model
model.fit(train_X, train_y, validation_data=(val_X, val_y), epochs=50, batch_size=32)

# Save model
model.save('cnn_lstm_energy_forecast.h5')


########## Model Evaluation ######################################

# Evaluate the model on test data
loss, mae = model.evaluate(test_X, test_y, verbose=1)
print(f"Test Loss: {loss}")
print(f"Test MAE: {mae}")

# Predict on the test data
predictions = model.predict(test_X)

print("Test values (first 10):", test_y[:10])
print("Predictions (first 10):", predictions[:10])

print("Test values range:", test_y.min(), test_y.max())
print("Predictions range:", predictions.min(), predictions.max())

import matplotlib.pyplot as plt

# Check if predictions and true values are the same shape
print(test_y.shape)
print(predictions.shape)

# Ensure that both are float64 for consistency
test_y = test_y.astype(np.float64)
predictions = predictions.astype(np.float64)

# Flatten predictions if needed (if they are 2D, e.g., (n_samples, 1))
predictions = predictions.flatten()

# Simple plot to check
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(test_y, label='True Values', color='blue')
ax.plot(predictions, label='Predictions', color='red', linestyle='--')
ax.set_title('True Values vs Predictions')
ax.set_xlabel('Sample Index')
ax.set_ylabel('Energy Generation')
ax.legend()
plt.show()


# Ensure plot renders
plt.show()

# Print the difference to check if it's small
print("Difference (first 10):", test_y[:10] - predictions[:10])

# Calculate additional performance metrics (Optional)
from sklearn.metrics import mean_squared_error
from math import sqrt

# Calculate RMSE (Root Mean Squared Error)
rmse = sqrt(mean_squared_error(test_y, predictions))
print(f"RMSE: {rmse}")

# Calculate R-squared (R² score)
from sklearn.metrics import r2_score
r2 = r2_score(test_y, predictions)
print(f"R-squared: {r2}")
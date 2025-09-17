import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime

# Load and preprocess data
file_path = "wn_route.csv"
df = pd.read_csv(file_path)

# Convert datetime and sort
df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d_%H%M')
df.sort_values('datetime', inplace=True)

# Extract values
timestamps = df['datetime'].values
data = df['pTC'].values.reshape(-1, 1)

# Normalize
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Create sequences
window_size = 30  # Increased window size
def create_sequences(data, timestamps, window_size):
    X, y, time_y = [], [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
        time_y.append(timestamps[i+window_size])
    return np.array(X), np.array(y), np.array(time_y)

X, y, time_y = create_sequences(data_scaled, timestamps, window_size)

# Split data into train/test
X_train_full, X_test, y_train_full, y_test, time_train_full, time_test = train_test_split(
    X, y, time_y, test_size=0.2, shuffle=False)

# Split train into train/validation
X_train, X_val, y_train, y_val, time_train, time_val = train_test_split(
    X_train_full, y_train_full, time_train_full, test_size=0.2, shuffle=False)

# Build model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(8, return_sequences=True, input_shape=(window_size, 1),
                         kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.LSTM(4, kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), loss='mse')

# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stop, lr_schedule],
    verbose=1
)

# Predict
y_pred = model.predict(X_test)
y_pred_rescaled = scaler.inverse_transform(y_pred)
y_test_rescaled = scaler.inverse_transform(y_test)

# RMSE
rmse = np.sqrt(mean_squared_error(y_test_rescaled, y_pred_rescaled))
print(f'Test RMSE: {rmse:.2f}')


import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# Read a CSV file into a DataFrame
df = pd.read_excel('2015_2024_Elec_Net_Gen_Data.xlsx', engine='openpyxl')

# Print the DataFrame
print(df)
df.head()

# Ensure all 'Net Generation' columns are numeric
gen_cols = [col for col in df.columns if 'Net Generation' in col]
df[gen_cols] = df[gen_cols].apply(pd.to_numeric, errors='coerce')

# Aggregate data at the state level
df_state = df.groupby(['Plant State', 'YEAR'])[gen_cols].sum().reset_index()
df_state.drop('Net Generation\nYear To Date', axis=1, inplace=True) # drop yearly aggregation column

# Convert wide format to long format
df_melted = df_state.melt(id_vars=['Plant State', 'YEAR'], 
                           var_name='Month', 
                           value_name='Net Generation')

# Extract actual month names
df_melted['Month'] = df_melted['Month'].str.replace('Net Generation\n', '')

# Convert month names to numerical format
df_melted['Month']
df_melted['Date'] = pd.to_datetime(df_melted['YEAR'].astype(str) + ' ' + df_melted['Month'], format='%Y %B')
df_melted = df_melted.sort_values(by=['Plant State', 'Date'])

# Drop unnecessary columns
df_melted = df_melted[['Plant State', 'Date', 'Net Generation']]

# Normalize data for LSTM input
scaler = MinMaxScaler()
df_melted['Net Generation'] = scaler.fit_transform(df_melted[['Net Generation']])

# Convert to TensorFlow-friendly time-series sequences
def create_sequences(data, state, sequence_length=12):
    state_data = data[data['Plant State'] == state].sort_values(by='Date')
    values = state_data['Net Generation'].values
    sequences = []
    targets = []
    for i in range(len(values) - sequence_length):
        sequences.append(values[i:i + sequence_length])
        targets.append(values[i + sequence_length])
    return np.array(sequences), np.array(targets)

# Create sequences for all states
sequence_length = 12
X, y = [], []
states = df_melted['Plant State'].unique()
for state in states:
    state_X, state_y = create_sequences(df_melted, state, sequence_length)
    X.append(state_X)
    y.append(state_y)

X = np.vstack(X)
y = np.concatenate(y)

# Split into training, validation, and test sets
train_size = int(len(X) * 0.7)
val_size = int(len(X) * 0.15)
X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

# Save preprocessed data
np.savez('preprocessed_energy_data.npz', X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)

print("Preprocessing complete. Data saved to 'preprocessed_energy_data.npz'")

# pickle the scaler
import pickle

# Save the fitted scaler for inverse transformation later
with open('net_gen_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Scaler saved as 'net_gen_scaler.pkl'")
import pandas as pd
import numpy as np
import pickle
from tensorflow.keras.models import load_model


# Load the scaler
with open('net_gen_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the trained CNN-LSTM model
model = load_model('cnn_lstm_energy_forecast.h5')

# Load and prepare monthly data
df_melted = pd.read_excel('2015_2024_Elec_Net_Gen_Data.xlsx', engine='openpyxl')
gen_cols = [col for col in df_melted.columns if 'Net Generation' in col]
df_melted[gen_cols] = df_melted[gen_cols].apply(pd.to_numeric, errors='coerce')
df_state = df_melted.groupby(['Plant State', 'YEAR'])[gen_cols].sum().reset_index()
df_state.drop('Net Generation\nYear To Date', axis=1, inplace=True)
df_melted = df_state.melt(id_vars=['Plant State', 'YEAR'], var_name='Month', value_name='Net Generation')
df_melted['Month'] = df_melted['Month'].str.replace('Net Generation\n', '')
df_melted['Date'] = pd.to_datetime(df_melted['YEAR'].astype(str) + ' ' + df_melted['Month'], format='%Y %B')
df_melted = df_melted.sort_values(by=['Plant State', 'Date'])
df_melted = df_melted[['Plant State', 'Date', 'Net Generation']]

# Normalize
df_melted['Net Generation'] = scaler.transform(df_melted[['Net Generation']])

# Forecast 60 months (5 years) per state
sequence_length = 12
forecast_horizon = 60
state_forecasts = []

for state in df_melted['Plant State'].unique():
    state_data = df_melted[df_melted['Plant State'] == state].sort_values(by='Date')
    values = state_data['Net Generation'].values

    if len(values) < sequence_length:
        print(f"Skipping state {state} due to insufficient data.")
        continue

    input_seq = values[-sequence_length:].tolist()
    last_date = state_data['Date'].max()

    for i in range(forecast_horizon):
        # Reshape to (batch, time_step, features)
        model_input = np.array(input_seq[-sequence_length:]).reshape((1, 1, sequence_length))
        pred_scaled = model.predict(model_input, verbose=0)[0][0]
        pred_actual = scaler.inverse_transform([[pred_scaled]])[0][0]

        forecast_date = last_date + pd.DateOffset(months=i + 1)
        state_forecasts.append({
            'Plant State': state,
            'Forecast Date': forecast_date,
            'Forecasted Net Generation': pred_actual
        })

        # Append the prediction (scaled) for the next loop
        input_seq.append(pred_scaled)

# Create final forecast DataFrame
forecast_df = pd.DataFrame(state_forecasts)
forecast_df['Month'] = forecast_df['Forecast Date'].dt.strftime('%B')
forecast_df['Year'] = forecast_df['Forecast Date'].dt.year

# Save
forecast_df.to_csv('state_forecasts_5yr_monthly.csv', index=False)
print("5-year monthly forecasts by state saved to 'state_forecasts_5yr_monthly.csv'.")
print(forecast_df.head())
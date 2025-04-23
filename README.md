# ML-Energy-Forecasting
Research project predicting energy demand using hybrid LSTM-CNN models

# Energy Generation Forecasting using CNN-LSTM
## Overview
This project focuses on building a hybrid deep learning model to forecast monthly energy generation by U.S. state. The model combines Convolutional Neural Networks (CNNs) for local pattern detection and Long Short-Term Memory (LSTM) layers for capturing long-term dependencies in the time series data.

##Problem Statement
Forecasting energy generation accurately is vital for managing power grid operations, energy trading, and long-term infrastructure planning. Traditional models like ARIMA treat energy generation as a stationary process, which may fail to capture complex temporal patterns. Our goal is to explore advanced deep learning approaches capable of making granular, state-level forecasts that account for regional variability.

## Dataset
The dataset is sourced from the U.S. Energy Information Administration (EIA) and includes:

Monthly net generation by plant

Metadata for plant state, generator ID, operator, fuel type, etc.

Time period: 2001–2023

## Preprocessing
Data grouped by Plant State and resampled monthly

Missing months filled with zero generation

Output shaped into 3D sequences: (num_samples, time_steps, features)

Scaled using MinMaxScaler for model training

## Model Architecture
CNN Layer: 1D convolution filters applied to extract short-term trends

MaxPooling: Downsamples the feature maps

LSTM Layer: Captures sequential dependencies

Dense Output Layer: Outputs predicted energy generation for the next month

Implemented using TensorFlow/Keras.

## Results
Achieved R² = 0.974 on national-level forecasts

Model generalizes well across multiple states, but performance varies due to volatility in data (e.g., Alaska and Wyoming)

Compared to ARIMA baseline, CNN-LSTM consistently outperformed in capturing nonlinear trends

## Visualizations
Time series plots comparing predicted vs actual values

Error metrics: MAE, RMSE, MAPE

Heatmaps showing per-state performance and volatility

## Conclusion
Accurately forecasting energy generation is a cornerstone of reliable power grid operation and long-term infrastructure planning. Traditional statistical models like ARIMA offer a baseline for prediction by treating energy demand as a stationary or slowly evolving time series. In many cases, this approach can yield reasonable forecasts, particularly when seasonal patterns and trends are stable.

However, as the energy landscape becomes increasingly dynamic—driven by renewable integration, policy shifts, and climate variability—stationarity-based models face significant limitations. This research highlights the value of adopting more expressive and adaptable models such as CNN-LSTM architectures. While ARIMA assumes linearity and requires manual differencing and parameter tuning, deep learning models can automatically learn nonlinear temporal dependencies and extract complex features directly from raw data.

In this study, the CNN layers efficiently captured short-term local patterns in the input sequences, while the LSTM layers modeled long-term dependencies, resulting in a highly accurate national-level forecast with an R² of 0.974. The high prediction variance (PM) observed in states like Alaska and Wyoming highlights regional volatility and emphasizes the importance of granularity in forecasting.

The impact of this modeling approach extends beyond immediate forecast accuracy. By demonstrating the effectiveness of hybrid deep learning models, this work encourages utility providers, policymakers, and researchers to explore advanced forecasting tools that are better equipped to handle the volatility and high dimensionality of modern energy systems.

## Future Work
- Model Enhancements: Explore deeper or wider CNN layers, add attention mechanisms, or integrate Transformer components into the LSTM pipeline

- Ensemble Approaches: Combine CNN-LSTM with GRUs or traditional models to improve robustness

- Granular Forecasts: Expand to state-level and plant-level forecasts for better localized decision-making

- Data Scope: Integrate policy indicators, weather data, and economic factors to further refine predictions

Ultimately, advancing this work involves not only refining model architectures but also expanding the scope of data inputs and temporal horizons, ensuring that forecasts remain both accurate and actionable in an evolving energy landscape.


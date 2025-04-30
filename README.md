# Energy Generation Forecasting using CNN-LSTM
![Visualization Image](https://github.com/user-attachments/assets/a40317d5-174c-4abf-9c36-775fae25d406)

## Overview

This project presents a machine learning-based framework for forecasting long-term energy demand across the United States. Using historical net generation data and a hybrid CNN-LSTM model, the study generates accurate monthly energy forecasts for each U.S. state and visualizes future demand using interactive choropleth maps. 

- Full research paper: Future Energy Needs

## Table of Contents

- Introduction

- Problem Statement

- Literature Review

- Proposed Method

- Data Preprocessing

- Time-Series Modeling

- CNN-LSTM Hybrid Architecture

- Visualization

- Evaluation

- Technologies Used

- Results

- Conclusion and Future Work

- How to Run

- Acknowledgements

## Introduction

With growing global energy demand, accurate energy load forecasting is essential for grid reliability, cost-effectiveness, and sustainable planning. This project aims to provide an adaptive and robust load forecasting solution that accounts for socio-economic, policy, and technological changes.

## Problem Statement

Forecasting energy generation accurately is vital for managing power grid operations, energy trading, and long-term infrastructure planning. Traditional models like ARIMA treat energy generation as a stationary process, which may fail to capture complex temporal patterns. Our goal is to explore advanced deep learning approaches capable of making granular, state-level forecasts that account for regional variability.

## Literature Review

### Market Factors

Key drivers include distributed generation, smart grid integration, regulatory shifts, and renewable energy adoption. Forecasting models must be region-specific and adapt to socio-economic and infrastructural differences.

### Machine Learning and Deep Learning Approaches

LSTM and CNN models provide high accuracy in temporal and nonlinear pattern detection. Hybrid methods combining ML with traditional models reduce forecasting errors.

### Statistical and Time Series Models

ARIMA and exponential smoothing models are still foundational, especially for short-term predictions.

### Feature Engineering

Feature extraction using PCA, wavelet transforms, and handling seasonality/missing values enhances model performance.

### Hybrid and Ensemble Methods

Combining statistical and ML methods boosts robustness and forecasting accuracy.

### Applications

Real-world applications emphasize weather, policy, and economic indicators as key forecasting inputs.

## Proposed Method

### Data Preprocessing

- Data sourced from the U.S. Energy Information Administration (EIA)

- Monthly net generation by plant with metadata (state, generator ID, operator, fuel type, etc.)

- Time period: 2015–2023

- Grouped by Plant State and resampled monthly

- Missing values filled with zero

- Validated for consistency

- Scaled using MinMaxScaler

### Time-Series Modeling

- Aggregated state-level monthly generation

- Used ARIMA model for univariate prediction

- Visual inspection via plots and autocorrelation functions

### CNN-LSTM Hybrid Architecture

- Net generation normalized (0–1)

- Input: past 12 months → Output: next month

- CNN layers extract short-term patterns

- MaxPooling for dimensionality reduction

- LSTM layers capture long-term dependencies

- Dropout layers to prevent overfitting

- Dense output layer with tanh activation

- Trained using Adam optimizer and MSE loss

## Visualization

- Time series plots comparing predicted vs actual values

- Choropleth maps built using D3.js

- Visualizes predicted energy demand by state

- Heatmaps show per-state performance and volatility

- Allows stakeholders to compare regional demand trends

## Evaluation

### ARIMA Model

- Training: 2015–2023, Test: 2024

- Metrics: MAPE, MAE, RMSE

- High accuracy in most states (MAPE < 1%)

### CNN-LSTM Model

- RMSE: 0.022

- R²: 0.974

- Outperforms ARIMA in capturing nonlinear and volatile trends

## Technologies Used

- Python (Pandas, NumPy, Scikit-learn, TensorFlow/Keras)

- R (ARIMA, forecast, dplyr, tidyr)

- D3.js for map visualizations

## Results

- Accurate 5-year forecasts per state

- CNN-LSTM generalizes well, despite volatility in states like Alaska and Wyoming

- Interactive choropleth map offers clear regional insights

- Hybrid model outperforms single-method approaches

## Conclusion and Future Work

Accurately forecasting energy generation is a cornerstone of reliable power grid operation and long-term infrastructure planning. Traditional statistical models like ARIMA offer a baseline for prediction by treating energy demand as a stationary or slowly evolving time series. In many cases, this approach can yield reasonable forecasts, particularly when seasonal patterns and trends are stable.

However, as the energy landscape becomes increasingly dynamic—driven by renewable integration, policy shifts, and climate variability—stationarity-based models face significant limitations. This research highlights the value of adopting more expressive and adaptable models such as CNN-LSTM architectures. While ARIMA assumes linearity and requires manual differencing and parameter tuning, deep learning models can automatically learn nonlinear temporal dependencies and extract complex features directly from raw data.

In this study, the CNN layers efficiently captured short-term local patterns in the input sequences, while the LSTM layers modeled long-term dependencies, resulting in a highly accurate national-level forecast with an R² of 0.974. The high prediction variance (PM) observed in states like Alaska and Wyoming highlights regional volatility and emphasizes the importance of granularity in forecasting.

The impact of this modeling approach extends beyond immediate forecast accuracy. By demonstrating the effectiveness of hybrid deep learning models, this work encourages utility providers, policymakers, and researchers to explore advanced forecasting tools that are better equipped to handle the volatility and high dimensionality of modern energy systems.

## Future Work

- Model Enhancements: Explore deeper or wider CNN layers, add attention mechanisms, or integrate Transformer components

- Ensemble Approaches: Combine CNN-LSTM with GRUs or traditional models

- Granular Forecasts: Expand to state-level and plant-level for better localized decision-making

- Data Scope: Integrate policy indicators, weather data, and economic factors

Ultimately, advancing this work involves not only refining model architectures but also expanding the scope of data inputs and temporal horizons, ensuring that forecasts remain both accurate and actionable in an evolving energy landscape.

## How to Run

For the CNN-LSTM model:
- LSTM_preprocessing.py: data preparation for model training
- CNN_LSTM_model.py: model training
- CNN_LSTM_forecast.py: forecasting using trained model over 5 year period

For the ARIMA models:
- modelGenerator.R: preprocessing, model training, and forecasting for 5 year period

For the visualization:
- maps.html: necessary libaries are stored within the CODE folder

A video demonstrating the visualizaiton can be accessed [here](https://youtu.be/K5vfsWBu56w).

## Acknowledgements

This research was conducted as part of energy forecasting efforts at Georgia Tech, supported by historical datasets from the U.S. Energy Information Administration (EIA).

# Project: Retail Sales Forecasting Project

## Description
This project focuses on forecasting weekly retail sales for individual stores using historical sales data. The dataset contains weekly sales data for multiple stores, and the objective is to predict future sales using advanced time series models like SARIMA. This enables businesses to optimize inventory, plan promotions, and make data-driven decisions.

## Business Use-case Overview
Retail stores often face challenges in inventory management, demand planning, and marketing. Accurate sales forecasting allows them to:

1.Reduce overstock and stockouts.
2.Allocate resources efficiently.
3.Identify trends and seasonal patterns.
4.Improve customer satisfaction by ensuring product availability.

Sales forecasting models like SARIMA capture trends, seasonality, and cyclic patterns in sales data, allowing retailers to predict sales for future weeks with high accuracy.

## Installation
This project requires Python 3.7+ and the following Python libraries:

numpy
pandas
matplotlib
seaborn
statsmodels
scikit-learn
streamlit

It is recommended to install these packages using pip or via the Anaconda distribution of Python.

## Data

The dataset consists of weekly sales data for multiple stores. It includes:

1.Store: Store ID (integer).
2.Date: Week ending date of the sales (YYYY-MM-DD).
3.Weekly_Sales: Total sales for the store in that week (float).
4.Other features (optional): Holiday flags etc.

The data is sourced from publicly available retail sales datasets, such as Kaggle’s Walmart Store Sales dataset

## Code
The project includes the following key scripts:

1.app.py: Streamlit app to visualize historical sales and generate forecasts.
2.Retail sales Forecasting.py: Script to train SARIMA models and generate forecasts.
3.utils/: Helper functions for preprocessing, plotting, and model evaluation.

The project allows users to interactively select a store, visualize past sales, and generate forecast graphs with trend lines.

## Run 
To run the project:
1.Navigate to the project folder
2.Run the Streamlit app:
    streamlit run app.py
3.Interact with the app:
Select a store from the sidebar.Apply filters for date range.Click Predict to generate.Forecasted sales graph with trend line.Forecast-only graph for clear visualization.

## Data Exploration
The dataset includes weekly sales for multiple stores. Initial exploration helps understand:
1.Trends over time.
2.Seasonal peaks (e.g., holidays, promotions).
3.Store-wise variation in sales.

## Forecasting Model
The project uses SARIMA (Seasonal ARIMA) to capture:
Trend: Overall increasing or decreasing sales over time.
Seasonality: Weekly, monthly, or yearly repeating patterns.
Noise: Random fluctuations in sales.

Model Training Steps:

1.Load and preprocess data.
2.Split into training and test sets.
3.Fit SARIMA model on training data.
4.Forecast future weeks and evaluate using MAE/R^2/RMSE.
5.Visualize forecasts alongside historical sales.

## Conclusion
The Retail Sales Forecasting project provides an end-to-end solution to predict future sales for retail stores. By analyzing historical weekly sales, the SARIMA model captures trends and seasonality, helping retailers make informed business decisions.

Key outcomes:

1.Predict weekly sales for multiple stores.
2.Visualize sales trends and seasonal patterns.
3.Generate actionable insights for inventory management and marketing.

Future Improvements:

1.Include additional features like promotions, holidays, and events.
2.Use more advanced models like Prophet or LSTM for better accuracy.
3.Extend the forecasting horizon beyond 12 weeks for long-term planning.
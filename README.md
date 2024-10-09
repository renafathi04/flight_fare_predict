# Flight Price Prediction Project

![project image](image/flight.webp)

## Project Overview

The Flight Price Prediction Project aims to develop a predictive model to forecast flight prices based on various features such as departure time, arrival time, airline, and other relevant factors. The goal is to assist travelers in making informed decisions when booking flights.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Data Description](#data-description)
4. [Model Training](#model-training)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Usage](#usage)
7. [License](#license)



### Objectives

- Analyze flight price data to identify trends and patterns.
- Train multiple regression models to predict flight prices.
- Evaluate model performance and select the best model based on RMSE or R² score.

## Installation

To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/flight-price-prediction.git
   cd flight-price-prediction
   ```

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

## Data Description

The dataset used in this project includes the following features:

- **Airline**: The airline operating the flight.
- **Date of Journey** 
- **Departure Time**: Scheduled departure time of the flight.
- **Arrival Time**: Scheduled arrival time of the flight.
- **Duration**: Total duration of the flight.
- **Source**: Departure city.
- **Destination**: Arrival city.
- **Total Stops**: number of stops 
- **Additional info**: Extra information
- **Price**: Ticket price for the flight.

## Model Training

This project involves training multiple regression models to predict flight prices:

1. **Linear Regression**
2. **Random Forest Regressor**
3. **Gradient Boosting Regressor**
4. **Decision Tree Regressor**
5. **XGBoost Regressor**

Each model is evaluated using cross-validation to ensure robust performance.

## Evaluation Metrics

The performance of the models is evaluated using the following metrics:

- **Root Mean Squared Error (RMSE)**: Measures the average magnitude of the errors.
- **R² Score**: Indicates the proportion of variance explained by the model.



## Usage

To use the model for predicting flight prices, follow these steps:

1. Load the trained model:
   ```python
   import joblib
   model = joblib.load('best_model.pkl')
   ```

2. Prepare your input data in the required format.

3. Make predictions:
   ```python
   predictions = model.predict(input_data)
   ```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

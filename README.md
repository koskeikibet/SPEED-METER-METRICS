# SPEED-METER-METRICS
Speed meter metrics, which measure the speed of various processes or systems, offer numerous benefits across a wide range of applications.
# Energy Consumption Prediction Using Random Forest

## Overview
This project uses a Random Forest Regressor model to predict energy consumption based on weather-related features such as temperature and humidity from a smart meter dataset. The dataset is based on electricity consumption data from India, and the model evaluates the predicted consumption against the actual consumption using Mean Squared Error (MSE).

## Project Components

### 1. **Libraries Used**
   - `pandas`: For data manipulation and analysis.
   - `matplotlib`: For visualizing the predicted vs actual consumption.
   - `sklearn.model_selection`: To split the dataset into training and testing sets.
   - `sklearn.ensemble.RandomForestRegressor`: The machine learning model used to predict energy consumption.
   - `sklearn.metrics`: To evaluate model performance using Mean Squared Error (MSE).

### 2. **Dataset**
   The dataset used in this project is an electricity smart meter dataset from India. The dataset includes features such as temperature, humidity, and energy consumption. It can be accessed from [Kaggle](https://www.kaggle.com/datasets/pythonafroz/electricity-smart-meter-data-from-india).

   **Note:** Make sure to update the dataset URL if necessary when downloading or using a local copy of the data.

### 3. **Data Preparation**
   - The data is loaded using the `pandas` library.
   - Features are selected based on available columns (e.g., temperature and humidity), and the target variable is energy consumption.

### 4. **Modeling**
   - The dataset is split into training and testing sets using an 80/20 split.
   - A `RandomForestRegressor` model is initialized and trained on the training data.
   - After training, the model is used to predict energy consumption on the test data.

### 5. **Evaluation**
   - Model performance is evaluated using Mean Squared Error (MSE) to measure how well the predicted energy consumption matches the actual data.

### 6. **Visualization**
   - A line plot is generated to compare actual vs predicted energy consumption, giving insights into the performance of the model visually.

## How to Run
1. **Install the necessary dependencies:**
   ```bash
   pip install pandas matplotlib scikit-learn


# Import necessary libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
# Import necessary libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score


import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_mail import Mail, Message
import io
import base64
import os
import random
import string
from reportlab.pdfgen import canvas
from flask import send_file
import tempfile










def perform_linear_regression(linear_input1, multiple_input1, multiple_input2, multiple_input3):
  # Load your dataset
  # Replace 'your_dataset.csv' with the actual file path or URL of your dataset
  # Load your dataset with a specified encoding
  # Replace 'your_dataset.csv' and 'your_encoding' with the actual file path and encoding of your dataset
  data = pd.read_csv('boxoffice.csv', encoding='ISO-8859-1')


  # Preprocess the data
  # Assuming you want to predict 'domestic_revenue' based on 'opening_revenue'

  # Handle missing values
  data = data.dropna()

  # Convert categorical variables if needed
  # Example: data['categorical_column'] = pd.Categorical(data['categorical_column']).codes

  # Extract features and target variable
  feature_column = 'opening_revenue'
  target_column = 'domestic_revenue'
  X = data[[feature_column]].copy()
  y = data[target_column].copy()


  X['opening_revenue'] = X['opening_revenue'].replace('[\$,]', '', regex=True).astype(float)
  y = y.replace('[\$,]', '', regex=True).astype(float)

  # Standardize or normalize numerical features
  scaler = StandardScaler(with_mean=False, with_std=True)
  X_scaled = scaler.fit_transform(X)

  # Split the data into training and testing sets
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

  # Train the model
  model = LinearRegression()
  model.fit(X_train, y_train)

  # Make predictions on the test set
  y_pred = model.predict(X_test)

  # Calculate Mean Squared Error (MSE)
  mse = mean_squared_error(y_test, y_pred)
  print(f'Mean Squared Error: {mse}')

  # Calculate R-squared
  r2 = r2_score(y_test, y_pred)
  print(f'R-squared: {r2}')



  # Make predictions on new data
  new_input_values = [[linear_input1]]
  new_input_scaled = scaler.transform(new_input_values)

  # Make predictions on new data
  new_predictions1 = model.predict(new_input_scaled)

  # Print or use the new predictions
  print(f'Predicted Domestic Revenue: ${new_predictions1[0]:,.2f}')






  # Load your dataset
  # Replace 'your_dataset.csv' with the actual file path or URL of your dataset
  # Load your dataset with a specified encoding
  # Replace 'your_dataset.csv' and 'your_encoding' with the actual file path and encoding of your dataset
  data = pd.read_csv('boxoffice.csv', encoding='ISO-8859-1')

  # Preprocess the data
  # Assuming you want to predict 'domestic_revenue' based on multiple features

  # Handle missing values
  data = data.dropna()

  # Convert categorical variables if needed
  # Example: data['categorical_column'] = pd.Categorical(data['categorical_column']).codes

  # Extract features and target variable
  feature_columns = ['opening_revenue', 'budget', 'release_days']  # Replace with your actual feature columns
  target_column = 'domestic_revenue'
  X = data[feature_columns].copy()
  y = data[target_column].copy()

  # Preprocess the features and target variable
  for column in feature_columns:
      X[column] = X[column].replace('[\$,]', '', regex=True).astype(float)

  y = y.replace('[\$,]', '', regex=True).astype(float)

  # Standardize or normalize numerical features
  scaler = StandardScaler(with_mean=False, with_std=True)
  X_scaled = scaler.fit_transform(X)

  # Split the data into training and testing sets
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

  # Train the model
  model = LinearRegression()
  model.fit(X_train, y_train)

  # Make predictions on the test set
  y_pred = model.predict(X_test)

  # Calculate Mean Squared Error (MSE)
  mse = mean_squared_error(y_test, y_pred)
  print(f'Mean Squared Error: {mse}')

  # Calculate R-squared
  r2 = r2_score(y_test, y_pred)
  print(f'R-squared: {r2}')
  # Get user input for prediction
  input_values = []

      # Convert the value to float
  input_values.append(float(multiple_input1))
  input_values.append(float(multiple_input2))
  input_values.append(float(multiple_input3))

  # Make predictions on new data
  new_input_values = [input_values]
  new_input_scaled = scaler.transform(new_input_values)

  # Make predictions on new data
  new_predictions2 = model.predict(new_input_scaled)

  # Print or use the new predictions
  print(f'Predicted Domestic Revenue: ${new_predictions2[0]:,.2f}')


  # Plot actual vs predicted values for Linear Regression
  plt.figure(figsize=(12, 6))

  # Actual values
  plt.scatter(X_test[:, 0], y_test, color='blue', label='Actual')

  # Predicted values
  plt.scatter(X_test[:, 0], y_pred, color='red', label='Linear Regression Prediction')

  plt.title('Linear Regression: Actual vs Predicted')
  plt.xlabel('Opening Revenue')
  plt.ylabel('Domestic Revenue')
  plt.legend()
  plt.show()

  # Calculate Mean Squared Error (MSE) and R-squared for Linear Regression
  linear_mse = mean_squared_error(y_test, y_pred)
  linear_r2 = r2_score(y_test, y_pred)
  print(f'Linear Regression Mean Squared Error: {linear_mse}')
  print(f'Linear Regression R-squared: {linear_r2}')

  # Plot actual vs predicted values for Multiple Regression
  plt.figure(figsize=(12, 6))

  # Actual values
  plt.scatter(X_test[:, 0], y_test, color='blue', label='Actual')

  # Predicted values
  multiple_y_pred = model.predict(X_test)
  plt.scatter(X_test[:, 0], multiple_y_pred, color='green', label='Multiple Regression Prediction')

  plt.title('Multiple Regression: Actual vs Predicted')
  plt.xlabel('Opening Revenue')
  plt.ylabel('Domestic Revenue')
  plt.legend()
  plt.show()

  # Calculate Mean Squared Error (MSE) and R-squared for Multiple Regression
  multiple_mse = mean_squared_error(y_test, multiple_y_pred)
  multiple_r2 = r2_score(y_test, multiple_y_pred)
  print(f'Multiple Regression Mean Squared Error: {multiple_mse}')
  print(f'Multiple Regression R-squared: {multiple_r2}')

  # Compare MSE and R-squared
  print('\nComparison:')
  print(f'Linear Regression Mean Squared Error: {linear_mse}')
  print(f'Multiple Regression Mean Squared Error: {multiple_mse}')
  print(f'Linear Regression R-squared: {linear_r2}')
  print(f'Multiple Regression R-squared: {multiple_r2}')

  # Choose the better model based on your preferred metric (lower MSE or higher R-squared)
  return new_predictions1[0],new_predictions2 [0]


app = Flask(__name__)

app.static_folder = 'static'



@app.route('/')
def index():
    return render_template('index.html')



@app.route('/predict', methods=['POST'])
def predict():
    value1 = request.form['feature1']
    value2 = request.form['feature2']
    value3 = request.form['feature3']
    value4 = request.form['feature4']
    result_linear , result_multiple = perform_linear_regression(value1,value2,value3,value4)
    output = f"The predicted domestic revenue is Rs.{result_linear:.2f}"
    output2 = f"The predicted domestic revenue is Rs.{result_multiple:.2f}"
  #  return render_template('result.html', prediction_text=output, prediction_text2=output2)
  #  return render_template('result.html', result_linear=result_linear, result_multiple=result_multiple)
    return redirect(f'/result?linear_result={output}&multiple_result={output2}')

@app.route('/result')
def result():
    # Get the results from the query parameters
    linear_result = request.args.get('linear_result')
    multiple_result = request.args.get('multiple_result')

    return render_template('result.html', linear_result=linear_result, multiple_result=multiple_result)







if __name__ == '__main__':
    app.run(debug=True)

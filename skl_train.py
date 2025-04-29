"""
    The following file trains the housing price prediction dataset using sklearn

    * Model trains and resets each time
"""
# Imports pandas, numpy, and sklearn modules 
import pandas as pd
import numpy as np
import sklearn as sk

# Imports classes and functions from the modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Allows the full dataframe to be displayed 
pd.set_option('display.max_columns', None)

# Loads housing price dataset into pandas dataframe (dataset)
dataset = pd.read_csv('housing_price_dataset.csv')

print(f"rows and columns: {dataset.shape}")

'''Prints the entire dataframe (house id, square feet, bedroom, bathroom, neighborhood, year built, price)
   print(dataset.to_string())'''

# Chooses the features and target 
features = ['SquareFeet', 'Bedrooms', 'Bathrooms', 'Neighborhood', 'YearBuilt']
target = ['Price']
X = dataset[features]
y = dataset[target]

# Transforms the data in the neighborhood column as binary values
x_encoded = pd.get_dummies(X, columns=['Neighborhood'], drop_first=False)
print(f"new columns after pd.get_dummies: {x_encoded.columns}")

# Splits the data for testing (20%) and training (80%)
X_train, X_test, y_train, y_test = train_test_split(x_encoded, y, test_size = 0.2, random_state = 42)

# Trains the dataset using linear regression and compares results using mean squared error
model = LinearRegression() # linear regression model object created
model.fit(X_train, y_train) # trains the data using the training data
predictions = model.predict(X_test) # predicts prices using the testing data
skl_mse = mean_squared_error(y_test, predictions) # compares predicted prices to actual prices
print(f"mean squared error: {skl_mse}") 

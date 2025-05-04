"""
    The following file trains the housing price prediction dataset using sklearn

    * Model trains and resets each time
"""
# Imports pandas, numpy, and sklearn modules 
import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt

# Imports classes and functions from the modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Optional: Neural network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

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

# Evaluation
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_r2 = r2_score(y_test, lr_predictions)
print(f"Linear Regression - MSE: {lr_mse:.2f}, R²: {lr_r2:.2f}")

#Visualizes predictions
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.6)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Housing Prices")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # ideal line
plt.show()

# ========== Neural Network Model (Optional) ==========
print("\n--- Training Neural Network (optional) ---")
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer
])

nn_model.compile(optimizer='adam', loss='mse')
nn_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

# Predict and evaluate
nn_predictions = nn_model.predict(X_test)
nn_mse = mean_squared_error(y_test, nn_predictions)
nn_r2 = r2_score(y_test, nn_predictions)
print(f"Neural Network - MSE: {nn_mse:.2f}, R²: {nn_r2:.2f}")

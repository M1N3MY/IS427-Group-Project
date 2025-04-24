"""
    The following file trains a model using sklearn
"""
# imports pandas, numpy, and sklearn modules 
import pandas as pd
import numpy as np
import sklearn as sk

# imports classes and functions from the modules
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# allows the full dataframe to be displayed 
pd.set_option('display.max_columns', None)

# loads housing price dataset into pandas dataframe (dataset)
dataset = pd.read_csv('housing_price_dataset.csv')

print(dataset.shape)

# prints the entire dataframe (house id, square feet, bedroom, bathroom, neighborhood, year built, price)
#print(dataset.to_string())

# chooses the features and target 
features = ['SquareFeet', 'Bedrooms', 'Bathrooms', 'Neighborhood', 'YearBuilt']
target = ['Price']
X = dataset[features]
y = dataset[target]

# transforms the data in the neighborhood column as binary values
dummies = pd.get_dummies(X, columns=['Neighborhood'], drop_first=False)
print(dummies.columns)
print(dummies)

print(dataset.corr()['price'].sort_values(ascending=False))

"""
mean squared error - loss, dividing by number of data points (n) 

generate linear regression model sci kit learn

improve error rate

normalization

"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle as pkl

# Read in data
df = pd.read_csv('Housing.csv')


model = LinearRegression()
model.fit(df[['area']], df['price'])


# Save the model using pickle
with open('linear_regression_model.pkl', 'wb') as file:
    pkl.dump(model, file)
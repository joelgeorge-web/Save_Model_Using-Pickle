import pickle as pkl
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


with open('linear_regression_model.pkl', 'rb') as file:
    model = pkl.load(file)

a = int(input("Enter the area: "))

# predict an output
print(model.predict([[a]]))
import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
#import a dataset
dataset = pd.read_csv('Data.csv')
independent = dataset.iloc[:, :-1].values
print(independent)
dependent = dataset.iloc[:, 3].values
print(dependent)
#Clear missing data
#  Imputer does fill the missing values with Average values
#fill the missing values with Average values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy="mean",axis = 0)
imputer = imputer.fit(independent[:, 1:3])
independent[:,1:3] = imputer.transform(independent[:, 1:3])
###
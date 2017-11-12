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
#Create Categorical variables
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label_independent = LabelEncoder()
independent[:, 0] = label_independent.fit_transform(independent[:, 0])
#Convert the Numerical category into three columns binary for model prediction
onehot_independent = OneHotEncoder(categorical_features=[0])
independent = onehot_independent.fit_transform(independent).toarray()
print("Independent after oneHot")
print(independent)
label_dependent = LabelEncoder()
dependent = label_dependent.fit_transform(dependent)
print(dependent)



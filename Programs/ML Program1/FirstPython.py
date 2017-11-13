import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
#import a dataset
dataset = pd.read_csv('Data.csv')
independent = dataset.iloc[:, :-1].values
print(independent)
dependent = dataset.iloc[:, 3].values
print(dependent)
print("*************************************************")
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
print("*************************************************")
#split test and train data
from sklearn.model_selection import  train_test_split
#returns a list of arrays. Read the array list and populate in different variables
independent_train,independent_test,dependent_train,dependent_test = train_test_split(independent,dependent,test_size=0.2,random_state=0)

print("TEST/TRAIN DATA")
print(dependent_test)
print("*************************************************")

#Apply Feature scaling on the Model to Standardise the data for easy calculations
from sklearn.preprocessing import StandardScaler
independent_SC = StandardScaler()
independent_train = independent_SC.fit_transform(independent_train) #fit the Standard scaler to the model and transform the model
independent_test = independent_SC.transform(independent_test)
print("*************************************************")
print(independent_train )
print(independent_test)
print("*************************************************")
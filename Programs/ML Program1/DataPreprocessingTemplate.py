import numpy as np
import matplotlib.pyplot as plot
import pandas as pd
#import a dataset
dataset = pd.read_csv('Programs/Data.csv')
independent = dataset.iloc[:, :-1].values
dependent = dataset.iloc[:, 3].values


#split test and train data
from sklearn.model_selection import  train_test_split
#returns a list of arrays. Read the array list and populate in different variables
independent_train,independent_test,dependent_train,dependent_test = train_test_split(independent,dependent,test_size=0.2,random_state=0)
print(dependent_test)


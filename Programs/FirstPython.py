import numpy as np
import matplotlib.pyplot as plot
import pandas as pd

dataset = pd.read_csv('Data.csv')
independent = dataset.iloc[:, :-1].values
print(independent)
dependent = dataset.iloc[:, 3].values
print(dependent)
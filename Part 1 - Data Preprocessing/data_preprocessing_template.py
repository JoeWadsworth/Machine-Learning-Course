# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

## Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
#Random state number is used just to match the same train and test data as the tutorial. Doesn't need to be imputted in reality.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Learns from training set. Trys out it's knowledge on the test set to see if worked correctly.

# Feature Scaling
#Put all columns on the same scale. 44 to 90 ~ -1 to +1
"""from sklearn.preprocessing import StandardScaler

#Fit to train first, then test.
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#Dont need to do it for y_test. Don't worry about it yet. changes in regression session.
# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
import statsmodels.api as sm

SL = 0.05

#Change constant in the function to a bunch of ones in the column
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X

while True:
    #Check P-Values
    regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
    #print(regressor_OLS.summary())

    #Convert regressor output to readable dataframe
    dfy = pd.read_html(regressor_OLS.summary().tables[1].as_html(),header=0)[0]
    dfy = dfy.iloc[:,[0,4]]
    dfy.columns = ['Variable','P-Value']
    
    #Remove constant
    dfy = dfy[dfy['Variable'] != 'const']
    
    #Find max P-Value
    max_val = dfy.max()[1]
    if max_val > SL:
        
        # Removes variable from model.
        col = dfy.loc[dfy['P-Value']==max_val].index[0]
        name = dfy[dfy['P-Value']==max_val]['Variable']
        print(f"Variable Removed: {col}, P-Value: {max_val}")
        X_opt = np.delete(X_opt,col,1)
    else: 
        print("No more variables.")
        print(regressor_OLS.summary())
        break
    
### Using SL
#import statsmodels.api as sm
#def backwardElimination(x, sl):
#    numVars = len(x[0])
#    for i in range(0, numVars):
#        regressor_OLS = sm.OLS(y, x).fit()
#        maxVar = max(regressor_OLS.pvalues).astype(float)
#        if maxVar > sl:
#            for j in range(0, numVars - i):
#                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
#                    x = np.delete(x, j, 1)
#    regressor_OLS.summary()
#    return x
# 
#SL = 0.05
#X_opt = X[:, [0, 1, 2, 3, 4, 5]]
#X_Modeled = backwardElimination(X_opt, SL)

### Using SL and R-Squared
#import statsmodels.api as sm
#def backwardElimination(x, SL):
#    numVars = len(x[0])
#    temp = np.zeros((50,6)).astype(int)
#    for i in range(0, numVars):
#        regressor_OLS = sm.OLS(y, x).fit()
#        maxVar = max(regressor_OLS.pvalues).astype(float)
#        adjR_before = regressor_OLS.rsquared_adj.astype(float)
#        if maxVar > SL:
#            for j in range(0, numVars - i):
#                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
#                    temp[:,j] = x[:, j]
#                    x = np.delete(x, j, 1)
#                    tmp_regressor = sm.OLS(y, x).fit()
#                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
#                    if (adjR_before >= adjR_after):
#                        x_rollback = np.hstack((x, temp[:,[0,j]]))
#                        x_rollback = np.delete(x_rollback, j, 1)
#                        print (regressor_OLS.summary())
#                        return x_rollback
#                    else:
#                        continue
#    regressor_OLS.summary()
#    return x
# 
#SL = 0.05
#X_opt = X[:, [0, 1, 2, 3, 4, 5]]
#X_Modeled = backwardElimination(X_opt, SL)
        
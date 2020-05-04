# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Encoding location (Spain, France, Germany etc.)
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

# Encoding Gender
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# Categorical variables arent ordinal. need to make dummy variables. only need to do location. 0,1,2 location (3) and 0,1 gender (2)
# Remove 1 dummy variable to avoid falling into the dummy variable trap. m categories, use m-1 in the model.
from sklearn.compose import make_column_transformer
ct = make_column_transformer((OneHotEncoder(categories='auto'), [1]),remainder="passthrough")
X = ct.fit_transform(X)
X = X[:, 1:]

# Split data to training set and test set.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Dense randomly generates the initial weights. 
# Need to chose activation function. sigmoid, rectifier etc.

# Creating Layer:
#  output_dim - number of nodes in layer.
#  init - values used for weight values. uniform in this case.
#  activation - activation function used. in this case rectifier function (relu).
#  input_dim - number of nodes in input layer.

## Initialising the ANN
classifier = Sequential()

## Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

## Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

## Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

## Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

## Fitting the ANN to the Training set
history = classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Plotting accuracy change of model over time
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

# Plotting loss change of model over time
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()

## Part 3 - Making the predictions and evaluating the model

## Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

## Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
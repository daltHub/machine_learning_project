import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statistics
import matplotlib as mpl
from sklearn.preprocessing import PolynomialFeatures
from sklearn import *
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression  # Import modules needed
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve, f1_score, confusion_matrix
from statistics import mode
from PIL import Image

df = pd.read_csv('clean_weather/valid_dates/m1_valid_dates.csv')
X1 = df.iloc[:, 5] # wind speed
X2 = df.iloc[:, 3] # atmospheric pressure
X = np.column_stack((X1, X2))
Y = df.iloc[:, 7] # wave height
p1 = plt.scatter(X1, X2, marker = 'x', c = Y, cmap = 'Accent')
# plt.xlabel('X1')
# plt.ylabel('X2')
# plt.show()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)  # split data
model1 = LinearRegression()
model1.fit(X_train, Y_train)
Y_pred = model1.predict(X_test)  # Predict on test data
p2 = plt.scatter(X_test[:,0], X_test[:, 1], marker = '.', c = Y_pred, cmap = 'tab20b')  # plot predictions
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('predictions')
plt.show()
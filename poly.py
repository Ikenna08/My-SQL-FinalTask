# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the csv_File
csv_File = pd.read_csv('NBA_Player_Stats.csv')
X = csv_File.iloc[:, 1:2].values
y = csv_File.iloc[:, 2].values

from sklearn.linear_model import LinearRegression
REG = LinearRegression()
REG.fit(X, y)

from sklearn.preprocessing import PolynomialFeatures
P_reg = PolynomialFeatures(degree = 7)
X_poly = P_reg.fit_transform(X)
P_reg.fit(X_poly, y)
LR2 = LinearRegression()
LR2.fit(X_poly, y)

# plot the Linear Regression results
plt.title('NBA Players')
plt.xlabel('Player game level')
plt.ylabel('assists')
plt.scatter(X, y, color = 'green')
plt.plot(X, REG.predict(X), color = 'maroon')
plt.show()

# plot the Polynomial Regression results
plt.title('NBA Players')
plt.xlabel('Player game level')
plt.ylabel('assists')
plt.scatter(X, y, color = 'red')
plt.plot(X, LR2.predict(P_reg.fit_transform(X)), color = 'purple')
plt.show()

# plot the Polynomial Regression results (for higher resolution and smoother curve)
X_Axis = np.arange(min(X), max(X), 0.1)
X_Axis = X_Axis.reshape((len(X_Axis), 1))
plt.ylabel('assists')
plt.title('NBA Players')
plt.xlabel('Player game level')
plt.scatter(X, y, color = 'grey')
plt.plot(X_Axis, LR2.predict(P_reg.fit_transform(X_Axis)), color = 'orange')
plt.show()

# Predicting a new plot with Linear Regression
REG.predict(3.7)

# Predicting a new plot with Polynomial Regression
LR2.predict(P_reg.fit_transform(3.7))

#https://www.geeksforgeeks.org/python-implementation-of-polynomial-regression/
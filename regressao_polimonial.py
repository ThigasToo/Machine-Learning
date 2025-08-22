# -*- coding: utf-8 -*-
"""Regressao_polimonial
Trying to fit different polynomial functions to the historical data of Bitcoin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import numpy as np

"""Checking data"""
dados = pd.read_csv('bit5.csv', sep=';')

dados.head()

X = dados['Data'].values
Y = dados['Último'].values

dados['Data'] = pd.to_datetime(dados['Data']) # Convert to datetime objects
X = dados['Data'].values
Y = dados['Último'].values

plt.scatter(X,Y, label='bit4')
plt.xlabel('Data')
plt.ylabel('Último')
plt.legend()
plt.show()

"""Regressions"""

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

"""Grau 2"""

caracteristicas_2 = PolynomialFeatures(degree=2)
X = X.reshape(-1,1)
X_Polinomio_2 = caracteristicas_2.fit_transform(X)

modelo2 = LinearRegression()
modelo2.fit(X_Polinomio_2, Y)
Y_Polinomio_2 = modelo2.predict(X_Polinomio_2)

plt.scatter(X,Y, label='bit5')
plt.plot(X, Y_Polinomio_2, color='red', label='Grau 2')
plt.xlabel('Data')
plt.ylabel('Último')
plt.legend()
plt.show()

"""Grau 3"""

caracteristicas_3 = PolynomialFeatures(degree=3)
X = X.reshape(-1,1)
X_Polinomio_3 = caracteristicas_3.fit_transform(X)

modelo3 = LinearRegression()
modelo3.fit(X_Polinomio_3, Y)
Y_Polinomio_3 = modelo3.predict(X_Polinomio_3)

plt.scatter(X,Y, label='bit5')
plt.plot(X, Y_Polinomio_3, color='red', label='Grau 3')
plt.xlabel('Data')
plt.ylabel('Último')
plt.legend()
plt.show()

"""Grau 4"""

caracteristicas_4 = PolynomialFeatures(degree=4)
X = X.reshape(-1,1)
X_Polinomio_4 = caracteristicas_4.fit_transform(X)

modelo4 = LinearRegression()
modelo4.fit(X_Polinomio_4, Y)
Y_Polinomio_4 = modelo4.predict(X_Polinomio_4)

plt.scatter(X,Y, label='bit5')
plt.plot(X, Y_Polinomio_4, color='red', label='Grau 4')
plt.xlabel('Data')
plt.ylabel('Último')
plt.legend()
plt.show()

"""Grau 5"""

caracteristicas_5 = PolynomialFeatures(degree=5)
X = X.reshape(-1,1)
X_Polinomio_5 = caracteristicas_5.fit_transform(X)

modelo5 = LinearRegression()
modelo5.fit(X_Polinomio_5, Y)
Y_Polinomio_5 = modelo5.predict(X_Polinomio_5)

plt.scatter(X,Y, label='bit5')
plt.plot(X, Y_Polinomio_5, color='red', label='Grau 5')
plt.xlabel('Data')
plt.ylabel('Último')
plt.legend()
plt.show()

"""Graus extremos"""

caracteristicas_x = PolynomialFeatures(degree=8)
X = X.reshape(-1,1)
X_Polinomio_x = caracteristicas_x.fit_transform(X)

modelox = LinearRegression()
modelox.fit(X_Polinomio_x, Y)
Y_Polinomio_x = modelox.predict(X_Polinomio_x)

plt.scatter(X,Y, label='bit5')
plt.plot(X, Y_Polinomio_x, color='red', label='Grau 8')
plt.xlabel('Data')
plt.ylabel('Último')
plt.legend()
plt.show()


"""Verifying the best fit through errors"""

from sklearn.metrics import mean_squared_error, mean_absolute_error

MAE2 = mean_absolute_error(Y, Y_Polinomio_2)
MAE3 = mean_absolute_error(Y, Y_Polinomio_3)
MAE4 = mean_absolute_error(Y, Y_Polinomio_4)
MAE5 = mean_absolute_error(Y, Y_Polinomio_5)
MAE8 = mean_absolute_error(Y, Y_Polinomio_x)

print('MAE Grau 2: ', MAE2)
print('MAE Grau 3: ', MAE3)
print('MAE Grau 4: ', MAE4)
print('MAE Grau 5: ', MAE5)
print('MAE Grau 8: ', MAE8)

RMSE2 = np.sqrt(mean_squared_error(Y, Y_Polinomio_2))
RMSE3 = np.sqrt(mean_squared_error(Y, Y_Polinomio_3))
RMSE4 = np.sqrt(mean_squared_error(Y, Y_Polinomio_4))
RMSE5 = np.sqrt(mean_squared_error(Y, Y_Polinomio_5))
RMSE8 = np.sqrt(mean_squared_error(Y, Y_Polinomio_x))

print('RMSE Grau 2: ', RMSE2)
print('RMSE Grau 3: ', RMSE3)
print('RMSE Grau 4: ', RMSE4)
print('RMSE Grau 5: ', RMSE5)
print('RMSE Grau 8: ', RMSE8)

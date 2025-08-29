# -*- coding: utf-8 -*-
"""LSTM-Bit2.ipynb
"""

!pip install tensorflow pandas seaborn matplotlib numpy scikit-learn

#imports
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os
from datetime import datetime



data = pd.read_csv('bit4.csv', sep=";")
data.head()

data.info()

data.describe()

plt.figure(figsize=(12, 6))
plt.plot(data['Data'], data['Abertura'], label='Open Price', color='blue')
plt.plot(data['Data'], data['Último'], label='Close Price', color='red')
plt.title('Open and Close Prices over Time')
plt.ylabel('Price')
plt.legend()
plt.show()

# Prepare for the LSTM Model (Sequential)
stock_close = data['Último']

dataset = stock_close.values
training_data_len = int(np.ceil(len(dataset) * 0.95))

# Preprocessing Stages
scaler = StandardScaler()
# Reshape the dataset to be a 2D array
dataset = dataset.reshape(-1, 1)
scaled_data = scaler.fit_transform(dataset)

training_data = scaled_data[:training_data_len]
X_train, y_train = [], []

# Create a sliding window for our stock
for i in range(60, len(training_data)):
    X_train.append(training_data[i-60:i, 0])
    y_train.append(training_data[i, 0])

plt.figure(figsize=(12, 6))
plt.plot(data['Data'], data['Último'], color='blue')
plt.title('Prices over Time')
plt.xlabel('Date')
plt.ylabel('Close')
plt.show()



X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#Modelo
model = keras.models.Sequential()
#First Layer
model.add(keras.layers.LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
#Second Layer
model.add(keras.layers.LSTM(units=64, return_sequences=False))
#3rd Layer
model.add(keras.layers.Dense(units=128, activation='relu'))
#4th Layer (dropout)
model.add(keras.layers.Dense(units=128, activation='relu'))
#Final Layer
model.add(keras.layers.Dense(units=1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error', metrics=[keras.metrics.RootMeanSquaredError])

training = model.fit(X_train, y_train, batch_size=32, epochs=50)

test_data = scaled_data[training_data_len - 60:]
X_test, y_test = [], dataset[training_data_len:]

for i in range(60, len(test_data)):
    X_test.append(test_data[i-60:i, 0])

X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#Prediction
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

#Ploting data
train = data[:training_data_len]
test = data[training_data_len:]

test = test.copy()

test['Predictions'] = predictions

plt.figure(figsize=(12, 8))
plt.plot(train['Data'], train['Último'], label='Train (Actual)', color='blue')
plt.plot(test['Data'], test['Último'], label='Test (actual)', color='green')
plt.plot( test['Data'], test['Predictions'],label='Predictions', color='red')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()
plt.show()

future_days = 30

# Últimos 60 valores do dataset escalado
last_60 = scaled_data[-60:]
future_input = last_60.reshape(1, -1, 1)

future_predictions = []

for _ in range(future_days):
    pred = model.predict(future_input)
    future_predictions.append(scaler.inverse_transform(pred)[0][0])

    # Atualiza a janela com a previsão
    new_input = np.append(future_input[0, 1:, 0], pred)
    future_input = new_input.reshape(1, -1, 1)

# Cria datas futuras a partir da última data conhecida
last_date = pd.to_datetime(data['Data'].iloc[-1])
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)

# DataFrame com as previsões futuras
future_df = pd.DataFrame({'Data': future_dates, 'Previsão_Futura': future_predictions})

# Plotando
plt.figure(figsize=(14, 8))
plt.plot(data['Data'], data['Último'], label='Fechamento Real', color='blue')
plt.plot(test['Data'], test['Predictions'], label='Previsões (teste)', color='red')
plt.plot(future_df['Data'].astype(str), future_df['Previsão_Futura'], label='Previsão Futura', color='orange', linestyle='--')
plt.title('Preço de Fechamento com Previsões Futuras')
plt.xlabel('Data')
plt.ylabel('Preço')
plt.legend()
plt.show()

# -*- coding: utf-8 -*-
"""sentiment_predict2.ipynb
A more complete and robust version of the last Sentiment Predict program
Just on the edge of overfitting
"""

!pip install kaggle

"""Importing the dependencies"""

import os
import json

from zipfile import ZipFile
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

"""Data Collection, Kaggle API"""

kaggle_dictionary = json.load(open('kaggle.json'))
kaggle_dictionary.keys()

#setup kaggle credentials as enviroment variables
os.environ['KAGGLE_USERNAME'] = kaggle_dictionary['username']
os.environ['KAGGLE_KEY'] = kaggle_dictionary['key']

!ls

#unzip the dataset file
with ZipFile('IMDB Dataset.csv.zip', 'r') as zip_ref:
  zip_ref.extractall()

!ls

"""Loading the dataset"""

data = pd.read_csv('IMDB Dataset.csv')
data.head()

data.info()

data.shape

#checking the data balance
data['sentiment'].value_counts()

#replacing the labels to numbers
data.replace({'sentiment': {'positive': 1, 'negative': 0}}, inplace=True)

data.head()

data['sentiment'].value_counts()

#split data into training and test data
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

print(train_data.shape)
print(test_data.shape)

"""Data Preprocessing"""

#Alteração para melhorar preprocessamento
#Clean the dataset
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    # remove HTML tags
    text = re.sub(r"<.*?>", " ", text)
    # remove pontuação e números
    text = re.sub(r"[^a-zA-Z]", " ", text)
    # lowercase
    text = text.lower()
    # remove stopwords
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# aplicar no dataset
data['review'] = data['review'].apply(clean_text)

# tokenize the data so the model can understand our reviews
tokenizer = Tokenizer(num_words=20000) #increase the numbers of words
tokenizer.fit_on_texts(train_data['review'])
X_train = pad_sequences(tokenizer.texts_to_sequences(train_data['review']), maxlen=200)
X_test = pad_sequences(tokenizer.texts_to_sequences(test_data['review']), maxlen=200)

print(X_train)

print(X_test)

Y_train = train_data['sentiment']
Y_test = test_data['sentiment']

print(Y_train)

print(Y_test)

"""LSTM: Long Short-Term Memory"""

# build the model

"""model = Sequential()
model.add(Embedding(5000, 128, input_length=200))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))"""

from tensorflow.keras.layers import Bidirectional, Dropout

model = Sequential()
model.add(Embedding(20000, 128, input_length=200))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(64)))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.build(input_shape=(None, 200))  # força a construção do modelo
model.summary()

model.summary()

# compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

"""Training the model"""

"""model.fit(X_train, Y_train, batch_size=64, epochs=5, validation_split=(0.2))"""

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss')

model.fit(X_train, Y_train,
          batch_size=64,
          epochs=15,   # aumenta épocas
          validation_split=0.2,
          callbacks=[early_stop, checkpoint])

"""Model avaluation"""

"""loss, accuracy = model.evaluate(X_test, Y_test)
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

y_pred = (model.predict(X_test) > 0.5).astype("int32")
print(classification_report(Y_test, y_pred))
print(confusion_matrix(Y_test, y_pred))
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')"""

loss, accuracy = model.evaluate(X_test, Y_test)

from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# gerar previsões
y_pred = (model.predict(X_test) > 0.5).astype("int32")

# classification_report espera arrays 1D
print(classification_report(Y_test, y_pred))

# confusion_matrix também espera arrays 1D
print(confusion_matrix(Y_test, y_pred))

print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

"""Building a predictive system

"""

def predict_sentiment(review):

  # tokenize and pad the review
  sequence = tokenizer.texts_to_sequences([review])
  paded_sequence = pad_sequences(sequence, maxlen=200)

  # make the prediction
  prediction = model.predict(paded_sequence)

  if prediction[0][0] > 0.7:
    sentiment = 'positive'
  elif prediction[0][0] < 0.3:
    sentiment = 'negative'
  else:
    sentiment = 'neutral'

  return sentiment

# example usage
new_review = 'This movie was fantastic. I loved every minute of it.'
sentiment = predict_sentiment(new_review)
print(f'The sentiment of the review is: {sentiment}')
print(model.predict(pad_sequences(tokenizer.texts_to_sequences([new_review]), maxlen=200)))

new_review = 'I liked it. There was times that the film was not great, but in all, it was a satisfying movie.'
sentiment = predict_sentiment(new_review)
print(f'The sentiment of the review is: {sentiment}')
print(model.predict(pad_sequences(tokenizer.texts_to_sequences([new_review]), maxlen=200)))

new_review = 'The movie was ok, but not good.'
sentiment = predict_sentiment(new_review)
print(f'The sentiment of the review is: {sentiment}')
print(model.predict(pad_sequences(tokenizer.texts_to_sequences([new_review]), maxlen=200)))

new_review = 'It was ok.'
sentiment = predict_sentiment(new_review)
print(f'The sentiment of the review is: {sentiment}')
print(model.predict(pad_sequences(tokenizer.texts_to_sequences([new_review]), maxlen=200)))

new_review = 'I did not care for it.'
sentiment = predict_sentiment(new_review)
print(f'The sentiment of the review is: {sentiment}')
print(model.predict(pad_sequences(tokenizer.texts_to_sequences([new_review]), maxlen=200)))

new_review = 'The movie had some interesting moments and a few strong performances, but it didn’t fully grab my attention. The pacing felt uneven, with certain scenes dragging while others felt rushed, with some really bad especial effects. Overall, it was an okay watch, but not something I’d revisit anytime soon.'
sentiment = predict_sentiment(new_review)
print(f'The sentiment of the review is: {sentiment}')
print(model.predict(pad_sequences(tokenizer.texts_to_sequences([new_review]), maxlen=200)))

# -*- coding: utf-8 -*-
"""Copy of sentiment_LSTM1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1teQtiOEREjyWOG_b4Vy2T4XcBFgdiWQY
"""

!pip install kaggle

from google.colab import drive
drive.mount('/content/drive')

"""**IMPORTING**"""

import os
import json

from zipfile import ZipFile
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

"""**DATA COLLECTION FROM KAGGLE**"""

kaggle_dictionary = json.load(open("kaggle.json"))

kaggle_dictionary.keys()

# setup kaggle credentials as environment variables
os.environ["KAGGLE_USERNAME"] = kaggle_dictionary["username"]
os.environ["KAGGLE_KEY"] = kaggle_dictionary["key"]

!kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

!ls

# unzip the dataset file
with ZipFile("imdb-dataset-of-50k-movie-reviews.zip", "r") as zip_ref:
  zip_ref.extractall()

!ls

"""**LOADING DATASET**"""

data = pd.read_csv("/content/IMDB Dataset.csv")

data.shape

data.head()

data["sentiment"].value_counts()

data.replace({"sentiment": {"positive": 1, "negative": 0}}, inplace=True)

data.head()

data["sentiment"].value_counts()

# split data into training data and test data
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

print(train_data.shape)
print(test_data.shape)

"""**DATA PREPROCESSING**"""

# Tokenize text data
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_data["review"])
X_train = pad_sequences(tokenizer.texts_to_sequences(train_data["review"]), maxlen=200)
X_test = pad_sequences(tokenizer.texts_to_sequences(test_data["review"]), maxlen=200)

print(X_train)

print(X_test)

Y_train = train_data["sentiment"]
Y_test = test_data["sentiment"]

print(Y_train)

"""**LSTM-LONG SHORT TERM MEMORY**"""

# build the model

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=200))
model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(1, activation="sigmoid"))

model.summary()

# compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

"""**TRANING**"""

model.fit(X_train, Y_train, epochs=10, batch_size=64, validation_split=0.2)

loss, accuracy = model.evaluate(X_test, Y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

"""**BUILDING A PREDICTIVE SYSTEM**"""

def predict_sentiment(review):
  # tokenize and pad the review
  sequence = tokenizer.texts_to_sequences([review])
  padded_sequence = pad_sequences(sequence, maxlen=200)
  prediction = model.predict(padded_sequence)
  sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
  return sentiment

# example usage
new_review = "This movie was fantastic. I loved it."
sentiment = predict_sentiment(new_review)
print(f"The sentiment of the review is: {sentiment}")

# example usage
new_review = "This movie was ok but not that good."
sentiment = predict_sentiment(new_review)
print(f"The sentiment of the review is: {sentiment}")

# example usage
new_review = "Peak Movie and best story line."
sentiment = predict_sentiment(new_review)
print(f"The sentiment of the review is: {sentiment}")


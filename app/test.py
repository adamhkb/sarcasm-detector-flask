import tensorflow as tf
from tensorflow import keras
from flask import Flask, render_template, request
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pickle

model = keras.models.load_model('model.h5')

maxlen = 100
text = ['Earthquake damage is caused by shaking']

tk = Tokenizer()
with open('tokenizer.pickle', 'rb') as handle:
    tk = pickle.load(handle)
X = tk.texts_to_sequences(text)
X = pad_sequences(X,maxlen=maxlen,padding='post',value=0)
pred = model.predict(X)
print(pred)
pred_perc = np.round(float(100 * pred), decimals=2)
print(pred_perc)
print(np.round(pred))

if np.round(pred) == 1:
    data = "Prediction:\n{} % sarcasm detected!\nTherefore, the headline is Sarcastic!".format(pred_perc)
else:
    data = "Prediction:\n{} % sarcasm detected!\nTherefore, the headline is not Sarcastic!".format(pred_perc)
print(data)
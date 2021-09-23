from flask import Flask, render_template, request
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import pickle

app = Flask(__name__)

def load_model():
    model = keras.models.load_model('model.h5')
    return (model)

@app.route('/')
def Home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    maxlen = 100
    text = request.form['Headline']
    tk = Tokenizer()
    with open('tokenizer.pickle', 'rb') as handle:
        tk = pickle.load(handle)
    X = tk.texts_to_sequences([text])
    X = pad_sequences(X,maxlen=maxlen,padding='post',value=0)
    model = load_model()
    pred = model.predict(X)
    pred_perc = np.round(float(100 * pred), decimals=2)
    data = "Prediction:"
    percent_text = "{}% sarcasm detected!".format(pred_perc)
    if np.round(pred) == 1:
        headline_text_results = "Therefore, the headline is Sarcastic!"
    else:
        headline_text_results = "Therefore, the headline is not Sarcastic!"
    return render_template('index.html',
                           headline_text = text,
                           prediction_text= data,
                           percent_text=percent_text,
                           headline_text_results=headline_text_results)

if __name__ == '__main__':
    app.run(debug=True)


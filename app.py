# -*- coding: utf-8 -*-
"""
Created on Fri May 22 06:08:33 2020

@author: Daniyal
"""


import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from flask import Flask, request, render_template

import nltk
nltk.data.path.append(r'C:\Users\Daniyal\Desktop\Important\NLP\nltk_data')
cv = pickle.load(open('transfrom.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=["POST"])
def predict():
    if request.method == "POST":
        message = request.form['message']
        review = []
        sent = [message]
        ps=PorterStemmer()
        sent = re.sub('[^A-Za-z]', ' ', message).lower().split()
        sent = [ps.stem(i) for i in sent if not i in stopwords.words('english')]
        sent = ' '.join(sent)
        review.append(sent)
        review = cv.transform(review).toarray()
        my_prediction = model.predict(review)
    return render_template('result.html', prediction=my_prediction)

if __name__ == '__main__':
    app.debug = True
    app.run()
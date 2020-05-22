# -*- coding: utf-8 -*-
"""
Created on Tue May 12 17:55:21 2020

@author: Daniyal
"""
import pandas as pd
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
data = pd.read_csv(r'C:\Users\Daniyal\Desktop\Important\Datasets\IMDBProcessed.csv', header=0)
x = list(data['review'])
y = np.array(data['sentiment'])
np.random.seed(1)

#%%
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features=50000)
x_train = cv.fit_transform(x)
y_train = np.array(y)
#%%
x_train_s, x_test_s, y_train, y_test = train_test_split(x_train,y_train,random_state=0)

#%%
modelPAC = PassiveAggressiveClassifier(early_stopping=True, C=0.1).fit(x_train_s, y_train)
#%%
modelLR = LogisticRegression(max_iter=2000).fit(x_train_s, y_train)
#%%
model = RandomForestClassifier(n_estimators=200, max_depth=18, min_samples_leaf=3, min_samples_split=3).fit(x_train_s, y_train)
#%%
modelNB = MultinomialNB().fit(x_train_s, y_train)
#%%
modelSVM = LinearSVC(C=0.05).fit(x_train_s, y_train)
#%%
#%%
print("Test Accuracies")
print("Random Forest        :     ", accuracy_score(model.predict(x_test_s), y_test).round(2))
print("Passive Agressive    :     ", accuracy_score(modelPAC.predict(x_test_s), y_test).round(2))
print("Logisitic Regression :     ", accuracy_score(modelLR.predict(x_test_s), y_test).round(2))
print("Naive Bayes          :     ", accuracy_score(modelNB.predict(x_test_s), y_test).round(2))
print("SVM                  :     ", accuracy_score(modelSVM.predict(x_test_s), y_test).round(2))

print("\nTrain Accuracies")
print("Random Forest        :     ", accuracy_score(model.predict(x_train_s), y_train).round(2))
print("Passive Agressive    :     ", accuracy_score(modelPAC.predict(x_train_s), y_train).round(2))
print("Logisitic Regression :     ", accuracy_score(modelLR.predict(x_train_s), y_train).round(2))
print("Naive Bayes          :     ", accuracy_score(modelNB.predict(x_train_s), y_train).round(2))
print("SVM                  :     ", accuracy_score(modelSVM.predict(x_train_s), y_train).round(2))
#%%
def predict(sent): 
    review = []
    ps=PorterStemmer()
    sent = re.sub('[^A-Za-z]', ' ', sent).lower().split()
    sent = [ps.stem(i) for i in sent if not i in stopwords.words('english')]
    sent = ' '.join(sent)
    review.append(sent)
    review = cv.transform(review).toarray()
    pred = modelSVM.predict(review)
    if pred==0:
        out = print('Seems like a POSITIVE review :)')
    else:
        out = print('Seems like a NEGATIVE review :(')
    return out
#%%
import pickle
pickle.dump(cv, open('transfrom.pkl', 'wb'))
pickle.dump(modelSVM, open('model.pkl', 'wb'))


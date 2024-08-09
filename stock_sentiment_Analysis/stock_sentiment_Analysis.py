#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:55:52 2024

@author: hrudaykumarkolla
"""

import pandas as pd

data = pd.read_csv('data.csv', sep=',', header=0, encoding='ISO-8859-1')
data['allnews'] = data.iloc[:, 2:].astype(str).apply(' '.join, axis=1)
data = data[['Date', 'Label', 'allnews']]

import re #regular expression module
import nltk #Natural language tool kit module
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

Data_processed_stemmed = []
Data_processed_lemmatized = []
for news in data['allnews']:
    news = re.sub('[^a-zA-Z]', ' ', news)
    news = news.lower()
    news = news.split()
    
    news_stemmed = [stemmer.stem(word) for word in news if not word in stopwords.words('english')]
    news_lemmatized = [lemmatizer.lemmatize(word) for word in news if not word in stopwords.words('english')]
    del news
    
    news_stemmed = ' '.join(news_stemmed)
    news_lemmatized = ' '.join(news_lemmatized)
    
    Data_processed_stemmed.append(news_stemmed)
    Data_processed_lemmatized.append(news_lemmatized)
    del news_stemmed
    del news_lemmatized

data['allnews_stemmed'] = Data_processed_stemmed
data['allnews_lemmatized'] = Data_processed_lemmatized


train_data = data[data['Date'] < '20150101'][['Date', 'Label', 'allnews_lemmatized']]
test_data = data[data['Date'] < '20141231'][['Date', 'Label', 'allnews_lemmatized']]
    
# bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000)
train_lemmatized_bow_features = cv.fit_transform(train_data['allnews_lemmatized']).toarray()
test_lemmatized_bow_features = cv.transform(test_data['allnews_lemmatized']).toarray()


from sklearn.ensemble import RandomForestClassifier

randomClassifier = RandomForestClassifier(n_estimators = 200, criterion = 'entropy')
randomClassifier.fit(train_lemmatized_bow_features,train_data['Label'])


predictions = randomClassifier.predict(test_lemmatized_bow_features)

## Import library to check accuracy
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score

matrix=confusion_matrix(test_data['Label'],predictions)

score=accuracy_score(test_data['Label'],predictions)

report=classification_report(test_data['Label'],predictions)





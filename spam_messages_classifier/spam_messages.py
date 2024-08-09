#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 12:52:00 2024

@author: hrudaykumarkolla
"""

import pandas as pd
import sklearn

messagesData = pd.read_csv('./sms+spam+collection/SMSSpamCollection', sep= '\t', header = None, names = ['labels', 'messages'])

bool_labels = pd.get_dummies(messagesData['labels'])
bool_labels_spam = bool_labels.iloc[:,1]

import re #regular expression module
import nltk #Natural language tool kit module
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

messagesData_processed_stemmed = []
messagesData_processed_lemmatized = []
for message in messagesData['messages']:
    message = re.sub('[^a-zA-Z]', ' ', message)
    message = message.lower()
    message = message.split()
    
    message_stemmed = [stemmer.stem(word) for word in message if not word in stopwords.words('english')]
    message_lemmatized = [lemmatizer.lemmatize(word) for word in message if not word in stopwords.words('english')]
    del message
    
    message_stemmed = ' '.join(message_stemmed)
    message_lemmatized = ' '.join(message_lemmatized)
    
    messagesData_processed_stemmed.append(message_stemmed)
    messagesData_processed_lemmatized.append(message_lemmatized)
    del message_stemmed
    del message_lemmatized
    
# bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000)
lemmatized_bow_features = cv.fit_transform(messagesData_processed_lemmatized).toarray()

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
lemmatized_tfidf_features = tfidf.fit_transform(messagesData_processed_lemmatized).toarray()

#spli data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(lemmatized_tfidf_features, bool_labels_spam, test_size = 0.20, random_state = 0)
    

#train model using Naive bayes classifier
from sklearn.naive_bayes import MultinomialNB
spam_model = MultinomialNB().fit(X_train, y_train)

y_pred = spam_model.predict(X_test)


conf_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)

accuracy_score = sklearn.metrics.accuracy_score(y_test, y_pred)
    


    
    
    
    
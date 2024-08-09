#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 12:15:58 2024

@author: hrudaykumarkolla
"""
# Fake News Classifier Using LSTM

# Dataset: https://www.kaggle.com/c/fake-news/data

import pandas as pd

data = pd.read_csv('train.csv')

'''
id: unique id for a news article
title: the title of a news article
author: author of the news article
text: the text of the article; could be incomplete
label: a label that marks the article as potentially unreliable
1: unreliable
0: reliable
'''

print(data.head())
print(data.describe())

# Counting NaN values in all columns
nan_count = data.isna().sum()

print(nan_count)

# Classifying only based on:
column_name = 'title'
data_used = data[[column_name, 'label']]
data_used = data_used.dropna()
data_used.reset_index(inplace=True)
x = data_used[[column_name]]
y = data_used[['label']]

#preprocess the text in titles
import re #regular expression module
import nltk #Natural language tool kit module
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

processed_titles = []
sent_size = []
for index, row in x.iterrows():
    title = re.sub('[^a-zA-Z]', ' ', row[column_name])
    title = title.lower()
    words_in_title = title.split()

    lemmatized_words_in_title = [lemmatizer.lemmatize(word) for word in words_in_title if not word in stopwords.words('english')]
    sent_size.append(len(lemmatized_words_in_title))
    lemmatized_title = ' '.join(lemmatized_words_in_title)
    processed_titles.append(lemmatized_title)

import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
#One hot representation
# Vocabulary size
voc_size=5000

onehot_repr=[one_hot(sentences,voc_size)for sentences in processed_titles]

sent_length= max(sent_size)+5
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
#print(embedded_docs)

## Creating model
embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

import numpy as np
X_final=np.array(embedded_docs)
y_final=np.array(y)
print(X_final.shape,y_final.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)

### Finally Training
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)

for th in range(11):
    y_pred=(model.predict(X_test) > (th/10)).astype("int32")
    
    from sklearn.metrics import confusion_matrix
    
    cf = confusion_matrix(y_test,y_pred)
    print(cf)
    from sklearn.metrics import accuracy_score
    print(accuracy_score(y_test,y_pred))

y_pred=(model.predict(X_test) > 0.5).astype("int32")

from sklearn.metrics import confusion_matrix

cf = confusion_matrix(y_test,y_pred)
print(cf)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))













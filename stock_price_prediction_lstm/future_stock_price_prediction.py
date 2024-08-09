#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 14:45:52 2024

@author: hrudaykumarkolla
"""

import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('AAPL-2.csv')
print(df.head())
#we will use only date and close price
df = df[['Date', 'Close']]
print(df.head())
#data is available until 4th April 2024
#We will do predictions for month of march
#using past 4 months of values, 120 days
future_pred_df = df[(df['Date'] >= '2024-03-01') & (df['Date'] <= '2024-03-30')]

train_test_df = df[(df['Date'] >= '2010-01-01') & (df['Date'] < '2024-03-01')]

train_test_df_index = train_test_df.reset_index()['Close']
plt.plot(train_test_df_index)
plt.show()

future_pred_df_index = future_pred_df.reset_index()['Close']


import numpy as np

from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
train_test_df_index_scaled=scaler.fit_transform(np.array(train_test_df_index).reshape(-1,1))
plt.plot(train_test_df_index_scaled)

##splitting dataset into train and test split
training_size=int(len(train_test_df_index_scaled)*0.65)
test_size=len(train_test_df_index_scaled)-training_size
train_data,test_data=train_test_df_index_scaled[0:training_size,:],train_test_df_index_scaled[training_size:len(train_test_df_index_scaled),:]


# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=100):
	dataX, dataY = [], []
	for i in range(len(dataset)-time_step-1):
		a = dataset[i:(i+time_step), 0]
		dataX.append(a)
		dataY.append(dataset[i + time_step, 0])
	return np.array(dataX), np.array(dataY)


time_step = 120
# X_train [samples, time steps]
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

# Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
model = None
model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(time_step,1)))
model.add(LSTM(50,return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error',optimizer='adam')

model.summary()

model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=100,batch_size=64,verbose=1)

import tensorflow as tf

### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)
test_predict=model.predict(X_test)

##Transformback to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)

### Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
train_rmse = math.sqrt(mean_squared_error(y_train,train_predict))
### Test Data RMSE
test_rmse = math.sqrt(mean_squared_error(ytest,test_predict))

### Plotting 
# shift train predictions for plotting
look_back=time_step
trainPredictPlot = np.empty_like(train_test_df_index_scaled)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(train_test_df_index_scaled)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(train_test_df_index_scaled)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(train_test_df_index_scaled))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


x_input=test_data[-time_step:].reshape(1,-1)

temp_input=list(x_input)
temp_input=temp_input[0].tolist()

# demonstrate prediction for next 30 days
from numpy import array

lst_output=[]
n_steps=time_step
i=0
while(i<len(future_pred_df_index)):
    
    if(len(temp_input)>time_step):
        #print(temp_input)
        x_input=np.array(temp_input[1:])
        print("{} day input {}".format(i,x_input))
        x_input=x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        lst_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        lst_output.extend(yhat.tolist())
        i=i+1
    

print(lst_output)

plt.plot(future_pred_df_index)
plt.plot(scaler.inverse_transform(lst_output))
plt.show()



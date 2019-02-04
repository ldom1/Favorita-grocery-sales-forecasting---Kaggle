#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 15:33:11 2019

@author: louisgiron
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import random as rd
from tqdm import tqdm
import pandas as pd
import os
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from seasonal import *
from statsmodels.tsa.tsatools import *
from statsmodels.tsa.stattools import *
from statsmodels.tsa.tsatools import detrend
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

from neural_network import neural_net_model, neural_net_model_3layers, rnn_model

# Refer to the py file: "processing_data.py"
from processing_data import preprocessing

n = None
store_selection = 'yes'

train_all, test = preprocessing(n, store_selection)

# Only with train yet
# We are studying only one store, so the information about the city, the
# region are useless

data = train_all.drop(['id', 'store_nbr', 'city', 'state',
                       'cluster', 'type_x'], axis=1)
data = data.set_index('date')

# Explore and manage the missing data
print('Before missing values management:')
print('----')
missing = data.apply(lambda x: x.isnull().sum(), axis=0)
missing = pd.DataFrame(missing)
missing.columns = ["Number of missing values"]

# Share of missing
missing['Share of missing'] = round(missing['Number of missing values']/len(data)*100,2)
print(missing)

missing_rate = {}
for col in data.columns:
    missing_rate[col] = np.sum(data[col].isna())/len(data[col])

missing_rate_lim = 0.7

for key in missing_rate.keys():
    if missing_rate[key] > missing_rate_lim:
        data = data.drop([key], axis=1)

    if 0 < missing_rate[key] < missing_rate_lim:
        # Handle missing values - Use the mean strategy
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(np.array(data[key]).reshape(-1, 1))
        temp_no_miss = imp.transform(np.array(data[key]).reshape(-1, 1))
        data[key] = temp_no_miss

print()
print('After missing values management:')
print('----')
# After the management
missing = data.apply(lambda x: x.isnull().sum(), axis=0)
missing = pd.DataFrame(missing)
missing.columns = ["Number of missing values"]

# Share of missing
missing['Share of missing'] = round(missing['Number of missing values']/len(data)*100,2)
print(missing)

# Categorical variables
# family is a category
family = data[['family', 'unit_sales']]
family = family.groupby(['family'], as_index=False)['unit_sales'].count()
family.columns = ['family', 'count']
family['repartition_percent'] = family['count']/np.sum(family['count'])*100
family = family.sort_values(by=['count'], ascending=False)
family.reset_index(inplace=True)
family = family.drop('index', axis=1)
print(family)


# Nous gardons les 3 premieres categories et créeons une catégorie other
def family_transform(x):
    if x in ['GROCERY I', 'BEVERAGES', 'CLEANING']:
        return x
    else:
        return 'other'


data['family'] = data['family'].apply(family_transform)
data = pd.get_dummies(data)

print(data.columns)

print(data.head(5))


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# load dataset
dataset = data.copy()
values = dataset.values

# ensure all data is float
values = values.astype('float32')

# frame as supervised learning
reframed = series_to_supervised(values, 1, 1)

# Visualisation
plt.figure()
plt.plot(list(dataset['unit_sales'])[500:600])
plt.title('Data original')
plt.show()

plt.figure()
plt.plot(reframed['var2(t-1)'][500:600])
plt.plot(reframed['var2(t)'][500:600])
plt.title('Data Reframed')
plt.show()

# split into train and test sets
values = reframed.values

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaler_fitted = scaler.fit(values)
scaled = scaler_fitted.transform(values)

# print scaled
print()
print('Scaled data')
print(scaled[1, ])
print()

n = int(0.6*reframed.shape[0])
train = scaled[:n, :]
test = scaled[n:, :]

# split into input and outputs
index_x = list(np.arange(train.shape[1]))
index_x.remove(1)

train_X, train_y = train[:, index_x], train[:, 1]

test_X, test_y = test[:, index_x], test[:, 1]

# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
print()


# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1],
               train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# fit network
history = model.fit(train_X, train_y, epochs=10, batch_size=250,
                    validation_data=(test_X, test_y), verbose=1, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('Evolution of the loss through epoch')
plt.xlabel("nombre d'epochs")
plt.ylabel("loss")
plt.legend()
plt.show()


# make a prediction
yhat = model.predict(test_X)

# Rescale the prediction between 0 and 1 - Under scaled predictions
scaler_pred = MinMaxScaler(feature_range=(0, 1))
scaler_fitted_pred = scaler_pred.fit(np.abs(yhat))
yhat_scaled = scaler_fitted_pred.transform(np.abs(yhat))

test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

# invert scaling for forecast
inv_yhat_temp = np.concatenate((yhat_scaled, test_X), axis=1)
inv_yhat_ = scaler_fitted.inverse_transform(inv_yhat_temp)
inv_yhat = inv_yhat_[:, 0]

# invert scaling for original
test_y = test_y.reshape((len(test_y), 1))
inv_y_temp = np.concatenate((test_y, test_X), axis=1)
inv_y_ = scaler_fitted.inverse_transform(inv_y_temp)
inv_y = inv_y_[:, 0]

# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)

# Resultat
plt.plot(test_y, label='Real data')
plt.title('Original data - Normed data')
plt.legend()
plt.show()

plt.plot(np.abs(yhat), label='predictions')
plt.title('Predictions - absolute values')
plt.legend()
plt.show()

plt.plot(yhat_scaled, label='predictions')
plt.title('Rescaled Predictions - absolute values')
plt.legend()
plt.show()

# Comparaison - Normed data
plt.plot(yhat_scaled, label='predictions')
plt.plot(test_y, label='Real data', alpha=0.6)
plt.title('Predictions - absolute values VS Original data - Normed data')
plt.legend()
plt.show()

# Comparaison - Non-scaled data
plt.plot(np.abs(inv_yhat), label='predictions')
plt.plot(inv_y, label='Real data', alpha=0.6)
plt.title('Predictions - absolute values VS Original data')
plt.legend()
plt.show()


window_size = np.arange(5, 50, 10)
nb_epoch = np.arange(5, 15, 5)
batch_size = np.arange(300, 500, 50)
j = 0
for size, epoch, batch in itertools.product(window_size, nb_epoch, batch_size):
    j += 1
print(j)

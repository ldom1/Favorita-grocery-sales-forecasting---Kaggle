#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 18 22:57:47 2018

@author: louisgiron
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import random as rd
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import f_regression

# Input data
path = os.getcwd() + '/all/'

train = pd.read_csv(path + 'train.csv', sep=',')
store = pd.read_csv(path + 'stores.csv', sep=',')
oil = pd.read_csv(path + 'oil.csv', sep=',')
holidays_event = pd.read_csv(path + 'holidays_events.csv', sep=',')
items = pd.read_csv(path + 'items.csv', sep=',')
transactions = pd.read_csv(path + 'transactions.csv', sep=',')

test = pd.read_csv(path + 'test.csv', sep=',')

# Construct the dataset
train_small = train

# Verify that unit_sales positive
train_small['unit_sales'] = train_small['unit_sales'].apply(lambda x: 0 if x <= 0 else x)

# Visualiser ventes par magasins
s_ = np.unique(train_small['store_nbr'])
s_ = s_[rd.randint(0, len(s_)-1)]
single_store = train_small[train_small['store_nbr'] == s_]
data_to_viz = single_store.groupby(['date', 'store_nbr'], as_index=False)['item_nbr'].count()
plt.plot(data_to_viz['date'], data_to_viz['item_nbr'])
plt.title('Vente - Store ' + str(s_))
plt.show()

# Create the training dataframe
# Merge store on store nb
train_all = train_small.merge(store, left_on='store_nbr',
                              right_on='store_nbr', how='left')
# Merge oil
train_all = train_all.merge(oil, left_on='date', right_on='date', how='left')

# Merge holiday
train_all = train_all.merge(holidays_event, left_on='date', right_on='date',
                            how='left')
# Merge items
train_all = train_all.merge(items, left_on='item_nbr', right_on='item_nbr',
                            how='left')
# Merge transactions
train_all = train_all.merge(transactions, left_on=['date', 'store_nbr'],
                            right_on=['date', 'store_nbr'], how='left')

# Explore and manage the missing data
missing = train_all.apply(lambda x: x.isnull().sum(), axis=0)
missing = pd.DataFrame(missing)
missing.columns = ["Number of missing values"]

# Share of missing
missing['Share of missing'] = round(missing['Number of missing values']/len(train_all)*100,2)
print(missing)

missing_rate = {}
for col in train_all.columns:
    missing_rate[col] = np.sum(train_all[col].isna())/len(train_all[col])

missing_rate_lim = 0.7

for key in missing_rate.keys():
    if missing_rate[key] > missing_rate_lim:
        train_all = train_all.drop([key], axis=1)

    if 0 < missing_rate[key] < missing_rate_lim:
        # Handle missing values - Use the mean strategy
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        imp.fit(np.array(train_all[key]).reshape(-1, 1))
        temp_no_miss = imp.transform(np.array(train_all[key]).reshape(-1, 1))
        train_all[key] = temp_no_miss

# Transform date to number
train_all['date'] = train_all['date'].apply(lambda x:
                                            int(''.join(i for i in x
                                                        if i.isdigit())))
# Manage category transformation
categorical_data = []
numerical_data = []

for col in train_all.columns:
    if train_all[col].dtype == 'O':
        categorical_data.append(col)
    else:
        numerical_data.append(col)

# Encode the categorical data
for category_col in categorical_data:
    train_all[category_col] = train_all[category_col].astype('category').cat.codes

# Feature importance
f_value, p_values = f_regression(train_all.drop(['unit_sales'], axis=1),
                                 np.array(train_all['unit_sales']),
                                 center=True)

# Plot p-value
plt.figure(figsize=(10, 4))
plt.bar(range(len(p_values)), p_values)
plt.axhline(y=0.05, color='r', linestyle='-')
plt.title('p-values for feature selection')
plt.xticks(range(len(p_values)), train_all.columns)
plt.show()

# We select all values

# Define X and y
selection = list(train_all.columns)
try:
    selection.remove('id')
except ValueError:
    pass

train_all = train_all[selection]

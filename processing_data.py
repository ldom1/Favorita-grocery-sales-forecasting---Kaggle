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


def preprocessing(n=None, store_selection=None):
    # Import of all the data
    path = os.getcwd() + '/all/'

    train = pd.read_csv(path + 'train.csv', sep=',')
    store = pd.read_csv(path + 'stores.csv', sep=',')
    oil = pd.read_csv(path + 'oil.csv', sep=',')
    holidays_event = pd.read_csv(path + 'holidays_events.csv', sep=',')
    items = pd.read_csv(path + 'items.csv', sep=',')
    transactions = pd.read_csv(path + 'transactions.csv', sep=',')

    test = pd.read_csv(path + 'test.csv', sep=',')

    # Construct the dataset
    if store_selection:
        store_unique = np.unique(train['store_nbr'])
        store_rd = np.random.randint(np.min(store_unique),
                                     np.max(store_unique), 1)
        train_small = train[train['store_nbr'] == int(store_rd)]
    else:
        if n:
            train_small = train[:n]
        else:
            train_small = train

    # Verify that unit_sales positive
    train_small['unit_sales'] = train_small['unit_sales'].apply(lambda x:
                                                                0 if x <= 0 else x)
    # Visualiser ventes par magasins
    s_ = np.unique(train_small['store_nbr'])
    s_ = s_[rd.randint(0, len(s_)-1)]
    single_store = train_small[train_small['store_nbr'] == s_]
    data_to_viz = single_store.groupby(['date', 'store_nbr'],
                                   as_index=False)['item_nbr'].count()
    plt.plot(range(len(data_to_viz['item_nbr'])), data_to_viz['item_nbr'])
    plt.xlabel('days')
    plt.ylabel('number of sales (total)')
    plt.title('Total number of sales - Store ' + str(s_))
    plt.show()

    # Create the training dataframe
    # Merge store on store nb
    train_all = train_small.merge(store, left_on='store_nbr',
                                  right_on='store_nbr', how='left')
    # Merge oil
    train_all = train_all.merge(oil, left_on='date', right_on='date',
                                how='left')

    # Merge holiday
    # issue of several causes for holiday
    holidays_event = holidays_event.drop_duplicates(subset='date')
    train_all = train_all.merge(holidays_event, left_on='date',
                                right_on='date', how='left')
    # Merge items
    train_all = train_all.merge(items, left_on='item_nbr', right_on='item_nbr',
                                how='left')
    # Merge transactions
    train_all = train_all.merge(transactions, left_on=['date', 'store_nbr'],
                                right_on=['date', 'store_nbr'], how='left')

    return train_all, test

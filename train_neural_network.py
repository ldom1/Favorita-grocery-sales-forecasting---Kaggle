#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 12:33:19 2018

@author: louisgiron
"""
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import random as rd
from tqdm import tqdm

from processing_data import train_all_processed
from neural_network import neural_net_model

# Train size
train_size = 0.6

# Split the train data to learn # Warning date -> split linearly
X_train = train_all_processed.drop(['unit_sales'],
                                   axis=1).loc[range(int(train_all_processed.shape[0]*train_size))]
y_train = train_all_processed['unit_sales'].loc[range(int(train_all_processed.shape[0]*train_size))]
X_test = train_all_processed.drop(['unit_sales'],
                                   axis=1).loc[range(int(train_all_processed.shape[0]*train_size),
                                               int(train_all_processed.shape[0]))]
y_test = train_all_processed['unit_sales'].loc[range(int(train_all_processed.shape[0]*train_size),
                                               int(train_all_processed.shape[0]))]

# Norlmalize data
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train.values)
y_train_norm = scaler.fit_transform(y_train.values.reshape(-1, 1))

X_test_norm = scaler.fit_transform(X_test.values)
y_test_norm = scaler.fit_transform(y_test.values.reshape(-1, 1))

print('X train shape:', X_train.shape)
print('X test shape:', X_test.shape)
print('Bounds for X train:')
print(np.max(X_train_norm), np.max(y_train_norm), np.min(y_train_norm),
      np.min(y_train_norm))


def denormalize(y_train, norm_data):
    try:
        df = y_train.values.reshape(-1, 1)
    except AttributeError:
        df = y_train.reshape(-1, 1)
    norm_data = norm_data.reshape(-1, 1)
    scl = MinMaxScaler()
    scl.fit_transform(df)
    return scl.inverse_transform(norm_data)


# Input data
nb_input = X_train.shape[1]
nb_hidden1 = 64
nb_hidden2 = 64
batch_size = 20000
nb_epoch = 30

# Initialize the model
X_tf = tf.placeholder(tf.float32)
y_tf = tf.placeholder(tf.float32)
keep_prob_1 = tf.placeholder(tf.float32)
keep_prob_2 = tf.placeholder(tf.float32)

output, W_O = neural_net_model(X_tf, nb_input, nb_hidden1, nb_hidden2,
                               keep_prob_1, keep_prob_2)

cost = tf.reduce_mean(tf.square(output-y_tf))
train = tf.train.AdamOptimizer(0.001).minimize(cost)

correct_pred = tf.argmax(output, 1)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

c_t = []
c_test = []
err_t = []  # norm l2
err_test = []

# Drop out level
prob_1 = 0.3
prob_2 = 0.3


# Training
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    y_t = denormalize(y_train, y_train_norm)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(1, 1, 1)
    plt.xlabel('item')
    plt.ylabel('number of units sales')
    plt.title('Evolution of units sales - Grocery store')
    ax.plot(range(len(y_train)), y_t, label='Original')
    plt.show()

    '''try:
        saver.restore(sess, 'NN_favorita_grocery_sales.ckpt')
    except Exception:
        pass'''

    for i in tqdm(range(nb_epoch)):

        # Define and create batch samples
        try:
            batch_start = rd.randint(0, X_train_norm.shape[0]-batch_size)
        except ValueError:
            batch_start = 0

        X_train_norm_batch = X_train_norm[np.arange(batch_start,
                                                    batch_start+batch_size),
                                          :]
        X_train_batch = X_train.loc[np.arange(batch_start,
                                              batch_start+batch_size)]

        y_train_norm_batch = y_train_norm[np.arange(batch_start,
                                                    batch_start+batch_size)].reshape(-1, 1)
        y_train_batch = y_train.loc[np.arange(batch_start,
                                              batch_start+batch_size)]
        y_train_batch = np.array(y_train_batch).reshape(-1, 1)

        # Run training on batch
        for j in range(X_train_norm_batch.shape[0]):
            sess.run([cost, train],
                     feed_dict={X_tf: X_train_norm_batch[j, :].reshape(1, nb_input),
                                y_tf: y_train_norm_batch[j],
                                keep_prob_1: prob_1,
                                keep_prob_2: prob_2})
        pred = sess.run(output, feed_dict={X_tf: X_train_norm_batch,
                                           keep_prob_1: 1.0,
                                           keep_prob_2: 1.0})
        pred = denormalize(y_train_batch, pred)

        # Compute the accuracy
        err_t.append(np.linalg.norm(pred - y_train_batch))
        c_t.append(sess.run(cost, feed_dict={X_tf: X_train_norm_batch,
                                             y_tf: y_train_norm_batch,
                                             keep_prob_1: 1.0,
                                             keep_prob_2: 1.0}))
        c_test.append(sess.run(cost, feed_dict={X_tf: X_test_norm,
                                                y_tf: y_test_norm,
                                                keep_prob_1: 1.0,
                                                keep_prob_2: 1.0}))
        print('Epoch :', i, 'Cost :', c_t[i], 'Err (l2) :', err_t[i])

    pred = sess.run(output, feed_dict={X_tf: X_test_norm,
                                       keep_prob_1: 1.0,
                                       keep_prob_2: 1.0})

    print('Cost :', sess.run(cost, feed_dict={X_tf: X_test_norm,
                                              y_tf: y_test_norm,
                                              keep_prob_1: 1.0,
                                              keep_prob_2: 1.0}))
    y_test = denormalize(y_test, y_test_norm)
    pred = denormalize(y_test, pred)

    # Plot the accuracy as l2 norm
    plt.figure(figsize=(10, 6))
    plt.plot(range(nb_epoch), err_t, label="err (l2 norm)")
    plt.legend(loc='best')
    plt.ylabel('l2 norm - error of the prediction')
    plt.xlabel('epochs')
    plt.title('Evolution of the error of the prediction through epochs')
    plt.show()

    # Plot the prediction vs the original
    plt.figure(figsize=(10, 6))
    plt.plot(range(y_test.shape[0]), y_test, label="Original Data")
    plt.plot(range(y_test.shape[0]), pred, label="Predicted Data")
    plt.legend(loc='best')
    plt.ylabel('units sales')
    plt.xlabel('Days')
    plt.title('Evolution of units sales - Grocery store')
    plt.show()

    # Save the model
    if input('Save model ? [Y/N]') == 'Y':
        import os
        saver.save(sess, os.getcwd() +
                   '/nn_saved_sessions/NN_favorita_grocery_sales.ckpt')
        print('Model Saved')

    # Close the session
    sess.close()

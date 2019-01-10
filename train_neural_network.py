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
from tqdm import tqdm

from processing_data import train_all_processed
from neural_network import neural_net_model, neural_net_model_3layers

# Train size
train_size_ratio = 0.5
valid_train_size_ratio = 0.1

size = train_all_processed.shape[0]

train_set = np.arange(0, int(train_size_ratio*size))
valid_set = np.arange(int(train_size_ratio*size), int(train_size_ratio*size)
                      + int(valid_train_size_ratio*size))
test_set = np.arange(int(train_size_ratio*size+valid_train_size_ratio*size),
                     int(size))

# Split the train data to learn # Warning date -> split linearly
X_train = train_all_processed.drop(['unit_sales'], axis=1).loc[train_set]
y_train = train_all_processed['unit_sales'].loc[train_set]
X_valid = train_all_processed.drop(['unit_sales'], axis=1).loc[valid_set]
y_valid = train_all_processed['unit_sales'].loc[valid_set]
X_test = train_all_processed.drop(['unit_sales'], axis=1).loc[test_set]
y_test = train_all_processed['unit_sales'].loc[test_set]

# Norlmalize data
scaler = MinMaxScaler()
X_train_norm = scaler.fit_transform(X_train.values)
y_train_norm = scaler.fit_transform(y_train.values.reshape(-1, 1))

X_valid_norm = scaler.fit_transform(X_valid.values)
y_valid_norm = scaler.fit_transform(y_valid.values.reshape(-1, 1))

X_test_norm = scaler.fit_transform(X_test.values)
y_test_norm = scaler.fit_transform(y_test.values.reshape(-1, 1))

print('X train shape:', X_train.shape)
print('X valid shape:', X_valid.shape)
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


""" 2 Layers Neural network """
# Input data
nb_epoch = 30
nb_input = X_train.shape[1]
nb_hidden1 = 12
nb_hidden2 = 8
batch_size = 1000

# Initialize the model
X_tf = tf.placeholder(tf.float32)
y_tf = tf.placeholder(tf.float32)
keep_prob_1 = tf.placeholder(tf.float32)
keep_prob_2 = tf.placeholder(tf.float32)

output = neural_net_model(X_tf, nb_input, nb_hidden1, nb_hidden2,
                          keep_prob_1, keep_prob_2)
avg_cost_v = []
c_t = []
c_valid = []
l_rate_v = []

# Drop out level
prob_1 = 0.6
prob_2 = 0.6

cost = tf.reduce_mean(tf.square(output-y_tf))
'''
# Piecewise constant
min_lr = 0.1
max_lr = 0.5
nb_values = nb_epoch

global_step = tf.Variable(0, trainable=False)
boundaries = list(np.linspace(batch_size,
                              batch_size*nb_epoch, nb_values,
                              dtype=np.int32)[:-1])
values = list(np.round(np.linspace(max_lr, min_lr, nb_values), 2))
learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

j = 0
# Passing global_step to minimize() will increment it at each step.
learning_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,
                                                                 global_step=global_step)'''

# Exponential decay of the learning rate
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.1
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           int(X_train.shape[0]/batch_size),
                                           0.96, staircase=True)
# Passing global_step to minimize() will increment it at each step.
learning_step = (
    tf.train.AdamOptimizer(learning_rate).minimize(cost,
                                                   global_step=global_step))
j = 0

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

    for epoch in tqdm(range(nb_epoch)):

        # Create the batches
        total_batch = int(X_train.shape[0]/batch_size)

        # Run training on batch
        for i in range(total_batch):
            avg_cost = 0.
            # Increment the learning rate with exponential decay
            l_rate = sess.run([learning_rate], {global_step: j})
            l_rate_v.append(l_rate)
            j += 1

            batch_X = X_train_norm[i*batch_size:(i+1)*batch_size]
            batch_y = y_train_norm[i*batch_size:(i+1)*batch_size]

            # Run the train
            _, c = sess.run([learning_step, cost],
                            feed_dict={X_tf: batch_X, y_tf: batch_y,
                                       keep_prob_1: prob_1,
                                       keep_prob_2: prob_2})
            # Compute average loss
            c_t.append(c)
            avg_cost += c / total_batch
        avg_cost_v.append(avg_cost)

        pred_norm, cost_valid = sess.run([output, cost],
                                         feed_dict={X_tf: X_valid_norm,
                                                    y_tf: y_valid_norm,
                                                    keep_prob_1: 1.0,
                                                    keep_prob_2: 1.0})
        c_valid.append(cost_valid)
        pred = denormalize(y_train, pred_norm)

        # Compute the accuracy
        print('Epoch :', epoch+1, 'Cost train:', avg_cost_v[epoch],
              'Cost valid:', c_valid[epoch], 'Learning rate:', l_rate)

    # Test the model
    print('Cost - Test phase:', sess.run(cost, feed_dict={X_tf: X_test_norm,
                                                          y_tf: y_test_norm,
                                                          keep_prob_1: 1.0,
                                                          keep_prob_2: 1.0}))
    pred_test_norm = sess.run(output, feed_dict={X_tf: X_test_norm,
                                                 keep_prob_1: 1.0,
                                                 keep_prob_2: 1.0})
    pred_test = denormalize(y_train, pred_test_norm)

    '''
    # Save the model
    if input('Save model ? [Y/N]') == 'Y':
        import os
        saver.save(sess, os.getcwd() +
                   '/nn_saved_sessions/NN_favorita_grocery_sales.ckpt')
        print('Model Saved')'''

    # Close the session
    sess.close()


""" 3 Layers Neural network """

# Input data
nb_epoch = 10
nb_input = X_train.shape[1]
nb_hidden1 = 16
nb_hidden2 = 128
nb_hidden3 = 32
batch_size = 5000

# Initialize the model
X_tf = tf.placeholder(tf.float32)
y_tf = tf.placeholder(tf.float32)
keep_prob_1 = tf.placeholder(tf.float32)
keep_prob_2 = tf.placeholder(tf.float32)
keep_prob_3 = tf.placeholder(tf.float32)

output = neural_net_model_3layers(X_tf, nb_input, nb_hidden1, nb_hidden2,
                                  nb_hidden3, keep_prob_1, keep_prob_2,
                                  keep_prob_3)
avg_cost_v = []
c_t = []
l_rate_v = []
c_valid = []

# Drop out level
prob_1 = 0.6
prob_2 = 0.4
prob_3 = 0.3

cost = tf.reduce_mean(tf.square(output-y_tf))
'''
# Piecewise constant
min_lr = 0.1
max_lr = 0.5
nb_values = nb_epoch

global_step = tf.Variable(0, trainable=False)
boundaries = list(np.linspace(batch_size,
                              batch_size*nb_epoch, nb_values,
                              dtype=np.int32)[:-1])
values = list(np.round(np.linspace(max_lr, min_lr, nb_values), 2))
learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

j = 0
# Passing global_step to minimize() will increment it at each step.
learning_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost,
                                                                 global_step=global_step)'''

# Exponential decay of the learning rate
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.1
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           int(X_train.shape[0]/batch_size),
                                           0.96, staircase=True)
# Passing global_step to minimize() will increment it at each step.
learning_step = (
    tf.train.AdamOptimizer(learning_rate).minimize(cost,
                                                   global_step=global_step))
j = 0

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

    for epoch in tqdm(range(nb_epoch)):

        # Create the batches
        total_batch = int(X_train.shape[0]/batch_size)

        # Run training on batch
        for i in range(total_batch):
            avg_cost = 0.
            # Increment the learning rate with exponential decay
            l_rate = sess.run([learning_rate], {global_step: j})
            l_rate_v.append(l_rate)
            j += 1

            batch_X = X_train_norm[i*batch_size:(i+1)*batch_size]
            batch_y = y_train_norm[i*batch_size:(i+1)*batch_size]

            # Run the train
            _, c = sess.run([learning_step, cost],
                            feed_dict={X_tf: batch_X, y_tf: batch_y,
                                       keep_prob_1: prob_1,
                                       keep_prob_2: prob_2,
                                       keep_prob_3: prob_3})
            # Compute average loss
            c_t.append(c)
            avg_cost += c / total_batch
        avg_cost_v.append(avg_cost)

        pred_norm, cost_valid = sess.run([output, cost],
                                         feed_dict={X_tf: X_valid_norm,
                                                    y_tf: y_valid_norm,
                                                    keep_prob_1: 1.0,
                                                    keep_prob_2: 1.0,
                                                    keep_prob_3: 1.0})
        c_valid.append(cost_valid)
        pred = denormalize(y_train, pred_norm)

        # Compute the accuracy
        print('Epoch :', epoch+1, 'Cost train:', avg_cost_v[epoch],
              'Cost valid:', c_valid[epoch], 'Learning rate:', l_rate)

    # Test the model
    print('Cost - Test phase:', sess.run(cost, feed_dict={X_tf: X_test_norm,
                                                          y_tf: y_test_norm,
                                                          keep_prob_1: 1.0,
                                                          keep_prob_2: 1.0,
                                                          keep_prob_3: 1.0}))
    pred_test = sess.run(output, feed_dict={X_tf: X_test_norm,
                                            keep_prob_1: 1.0,
                                            keep_prob_2: 1.0,
                                            keep_prob_3: 1.0})
    pred_test = denormalize(y_test, pred_test)

    '''
    # Save the model
    if input('Save model ? [Y/N]') == 'Y':
        import os
        saver.save(sess, os.getcwd() +
                   '/nn_saved_sessions/NN_favorita_grocery_sales.ckpt')
        print('Model Saved')'''

    # Close the session
    sess.close()

# Plot the cost of training
fig, ax1 = plt.subplots(figsize=(10, 7))
x = np.arange(nb_epoch)
ax1.plot(x, avg_cost_v, 'b-')
ax1.set_xlabel('epochs')
# Make the y-axis label and tick labels match the line color.
ax1.set_ylabel('Avg Cost train', color='b')
for tl in ax1.get_yticklabels():
    tl.set_color('b')


ax2 = ax1.twinx()
ax2.plot(x, c_valid, 'r-')
ax2.set_ylabel('Cost valid', color='r')
for tl in ax2.get_yticklabels():
    tl.set_color('r')
plt.show()

# Plot the learning rate
plt.figure(figsize=(10, 6))
plt.plot(l_rate_v, label="learning rate")
plt.legend(loc='best')
plt.ylabel('learning rate')
plt.xlabel('epochs')
plt.title('Evolution of the learning through epochs')
plt.show()

# Plot the prediction vs the original
plt.figure(figsize=(10, 6))
plt.plot(range(y_test.shape[0]), y_test, label="Original Data")
plt.plot(range(y_test.shape[0]), pred_test, label="Predicted Data")
plt.legend(loc='best')
plt.ylabel('units sales')
plt.xlabel('Days')
plt.title('Evolution of units sales - Grocery store')
plt.show()

# Plot the prediction
plt.figure(figsize=(10, 6))
plt.plot(range(y_test.shape[0]), pred_test, label="Predicted Data")
plt.legend(loc='best')
plt.ylabel('units sales')
plt.xlabel('Days')
plt.title('Predictions of units sales - Grocery store')
plt.show()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 23:41:07 2019

@author: louisgiron
"""
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tqdm import tqdm

from processing_data import train_all_processed
from neural_network import rnn_model

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


nb_inputs = X_train.shape[0]
num_components = X_train.shape[1]

nb_epoch = 50
num_hidden = 32
learning_rate = 0.001
lambda_loss = 0.001
batch_size = 1000

acc_t = []
loss_t = []
acc_val = []
loss_val = []


def accuracy(y_predicted, y):
    return (100.0 * np.sum(np.argmax(y_predicted, 1) == np.argmax(y, 1))
            / y_predicted.shape[0])


####
graph = tf.Graph()
with graph.as_default():
    # 1) First we put the input data in a tensorflow friendly form.
    X_tf = tf.placeholder(tf.float32, shape=(None, nb_inputs,
                                             num_components))
    y_tf = tf.placeholder(tf.float32)

    # 2) Then we choose the model to calculate the logits (predicted labels)
    # We can choose from several models:
    logits = rnn_model(X_tf, num_hidden, y_tf)
    # logits = lstm_rnn_model(tf_dataset, num_hidden, num_labels)

    # 3) Then we compute the softmax cross entropy between the logits and
    # the (actual) labels
    l2 = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_tf)) + l2

    # 4)
    # The optimizer is used to calculate the gradients of the loss function
    # optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    # Predictions for the training, validation, and test data.
    prediction = tf.nn.softmax(logits)


with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()
    print("\nInitialized")

    for epoch in tqdm(range(nb_epoch)):

        # Create the batches
        total_batch = int(X_train.shape[0]/batch_size)

        # Run training on batch
        for i in range(total_batch):
            # Since we are using stochastic gradient descent, we are selecting
            # small batches from the training dataset, and training the
            # convolutional neural network each time with a batch.
            batch_X = X_train_norm[i*batch_size:(i+1)*batch_size]
            batch_y = y_train_norm[i*batch_size:(i+1)*batch_size]

            feed_dict = {X_tf: batch_X, y_tf: batch_y}
            _, l, train_predictions = sess.run([optimizer, loss, prediction],
                                               feed_dict=feed_dict)
            train_accuracy = accuracy(train_predictions, batch_y)
            acc_t.append(train_accuracy)
            loss_t.append(l)

        if epoch % 10 == 0:
            feed_dict = {X_tf: X_valid_norm, y_tf: y_valid_norm}
            l_val, valid_predictions = sess.run([loss, prediction],
                                                feed_dict=feed_dict)
            valid_accuracy = accuracy(valid_predictions, y_valid_norm)
            print('Epoch ', str(epoch), ': ', ' - Accuracy on train set:',
                  train_accuracy, ' - Accuract on valid set:',  valid_accuracy)

            acc_val.append(valid_accuracy)
            loss_val.append(l_val)

    # Test set
    feed_dict = {X_tf: X_test_norm, y_tf: y_test_norm}
    _, test_predictions = sess.run([loss, prediction], feed_dict=feed_dict)
    valid_accuracy = accuracy(test_predictions, y_test_norm)
    print('Accuracy on test set:', valid_accuracy)

    # Close the session
    sess.close()

# Denormalize
pred_test = denormalize(y_test, test_predictions)

# Plot the prediction vs the original
plt.figure(figsize=(10, 6))
plt.plot(range(y_test.shape[0]), y_test, label="Original Data")
plt.plot(range(y_test.shape[0]), pred_test, label="Predicted Data")
plt.legend(loc='best')
plt.ylabel('units sales')
plt.xlabel('items')
plt.title('Evolution of units sales - Grocery store')
plt.show()

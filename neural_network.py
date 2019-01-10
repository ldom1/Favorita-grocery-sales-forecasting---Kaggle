#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 12:32:00 2018

@author: louisgiron
"""
import tensorflow as tf


# Define the graph - Basic neural net
def neural_net_model(X_data, nb_input, nb_hidden1, nb_hidden2, keep_prob_1,
                     keep_prob_2):

    # Layer 1
    W_1 = tf.Variable(tf.random.normal((nb_input, nb_hidden1),
                                       mean=0.0, stddev=1.0,
                                       dtype=tf.float32,))
    b_1 = tf.Variable(tf.zeros([nb_hidden1]))
    layer_1 = tf.add(tf.matmul(X_data, W_1), b_1)
    layer_1 = tf.nn.relu(layer_1)
    # apply DropOut to hidden layer 1
    drop_out_1 = tf.nn.dropout(layer_1, keep_prob_1)  # DROP-OUT here

    # Layer 2
    W_2 = tf.Variable(tf.random.normal((nb_hidden1, nb_hidden2),
                                       mean=0.0, stddev=1.0,
                                       dtype=tf.float32,))
    b_2 = tf.Variable(tf.zeros([nb_hidden2]))
    layer_2 = tf.add(tf.matmul(drop_out_1, W_2), b_2)
    layer_2 = tf.nn.relu(layer_2)
    # apply DropOut to hidden layer 2
    drop_out_2 = tf.nn.dropout(layer_2, keep_prob_2)  # DROP-OUT here

    # Output layer
    W_out = tf.Variable(tf.random.normal((nb_hidden2, 1),
                                         mean=0.0, stddev=1.0,
                                         dtype=tf.float32,))
    b_out = tf.Variable(tf.zeros([1]))
    output = tf.add(tf.matmul(drop_out_2, W_out), b_out)

    return output


def neural_net_model_3layers(X_data, nb_input, nb_hidden1, nb_hidden2,
                             nb_hidden3, keep_prob_1, keep_prob_2,
                             keep_prob_3):

    # Layer 1
    W_1 = tf.Variable(tf.random.normal((nb_input, nb_hidden1),
                                       mean=0.0, stddev=1.0,
                                       dtype=tf.float32,))
    b_1 = tf.Variable(tf.zeros([nb_hidden1]))
    layer_1 = tf.add(tf.matmul(X_data, W_1), b_1)
    layer_1 = tf.nn.relu(layer_1)
    # apply DropOut to hidden layer 1
    drop_out_1 = tf.nn.dropout(layer_1, keep_prob_1)  # DROP-OUT here

    # Layer 2
    W_2 = tf.Variable(tf.random.normal((nb_hidden1, nb_hidden2),
                                       mean=0.0, stddev=1.0,
                                       dtype=tf.float32,))
    b_2 = tf.Variable(tf.zeros([nb_hidden2]))
    layer_2 = tf.add(tf.matmul(drop_out_1, W_2), b_2)
    layer_2 = tf.nn.relu(layer_2)
    # apply DropOut to hidden layer 2
    drop_out_2 = tf.nn.dropout(layer_2, keep_prob_2)  # DROP-OUT here

    # Layer 3
    W_3 = tf.Variable(tf.random.normal((nb_hidden2, nb_hidden3),
                                       mean=0.0, stddev=1.0,
                                       dtype=tf.float32,))
    b_3 = tf.Variable(tf.zeros([nb_hidden3]))
    layer_3 = tf.add(tf.matmul(drop_out_2, W_3), b_3)
    layer_3 = tf.nn.relu(layer_3)
    # apply DropOut to hidden layer 2
    drop_out_3 = tf.nn.dropout(layer_3, keep_prob_3)  # DROP-OUT here

    # Output layer
    W_out = tf.Variable(tf.random.normal((nb_hidden3, 1),
                                         mean=0.0, stddev=1.0,
                                         dtype=tf.float32,))
    b_out = tf.Variable(tf.zeros([1]))
    output = tf.add(tf.matmul(drop_out_3, W_out), b_out)

    return output


# Define the graph - RNN neural net
def rnn_model(X_data, num_hidden):
    splitted_data = tf.unstack(X_data, axis=1)

    cell = tf.nn.rnn_cell.BasicRNNCell(num_hidden)

    outputs, current_state = tf.nn.static_rnn(cell, splitted_data,
                                              dtype=tf.float32)
    output = outputs[-1]
    w_softmax = tf.Variable(tf.truncated_normal([num_hidden, 1]))
    b_softmax = tf.Variable(tf.random_normal([1]))
    logit = tf.matmul(output, w_softmax) + b_softmax
    return logit

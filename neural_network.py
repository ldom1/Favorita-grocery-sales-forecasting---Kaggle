#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 12:32:00 2018

@author: louisgiron
"""
import tensorflow as tf


# Define the graph
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
    W_O = tf.Variable(tf.random.normal((nb_hidden2, 1),
                                       mean=0.0, stddev=1.0,
                                       dtype=tf.float32,))
    b_O = tf.Variable(tf.zeros([1]))
    output = tf.add(tf.matmul(drop_out_2, W_O), b_O)

    return output, W_O

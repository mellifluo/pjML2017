#!/usr/bin/env python

import tensorflow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# load dataset
X = np.loadtxt('./monk/monks-1.train', dtype='string', delimiter=' ')
# label data
y = X[:,1].astype('float32')
y = y[..., np.newaxis]
# input data
X = X[:,2:-1].astype('float32')
# settings
inputlayer_neurons = int(X.shape[1])
hiddenlayer_neurons = 3
output_neurons = 1
epoch = 100
lr = 0.2
beta = 0.01

graph = tf.Graph()
with graph.as_default():
    # input
    a_0 = tf.constant(X)
    # output
    output = tf.constant(y)
    # weight and bias initialization
    # hidden layer
    w_1 = tf.Variable(tf.random_uniform([inputlayer_neurons, hiddenlayer_neurons], minval=0.1 , maxval=0.9 , dtype=tf.float32))
    b_1 = tf.Variable(tf.random_uniform([1, hiddenlayer_neurons], minval=0.1 , maxval=0.9 , dtype=tf.float32))
    # output layer
    w_2 = tf.Variable(tf.random_uniform([hiddenlayer_neurons, output_neurons], minval=0.1 , maxval=0.9 , dtype=tf.float32))
    b_2 = tf.Variable(tf.random_uniform([1, output_neurons], minval=0.1 , maxval=0.9 , dtype=tf.float32))
    # putting together
    a_1 = tf.nn.sigmoid(tf.add(tf.matmul(a_0, w_1), b_1))
    a_2 = tf.nn.sigmoid(tf.add(tf.matmul(a_1, w_2), b_2))
    # error and backprop
    loss = tf.subtract(output, a_2)
    step = tf.train.GradientDescentOptimizer(lr).minimize(loss)

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    # useful for visualization
    writer = tf.summary.FileWriter("output", sess.graph)
    # training
    for i in range(epoch):
        _ = sess.run(step)
    writer.close()
    # ancora non ho capito bene qui, error da risultati strani
    # ammà chiedere a federico
    error = sess.run(loss)
    result = a_2.eval()

plt.plot(y)
plt.plot(error)

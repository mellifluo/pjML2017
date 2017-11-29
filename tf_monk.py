#!/usr/bin/env python

import tensorflow
import tensorflow as tf
import numpy as np

# load dataset
X = np.loadtxt('./monk/monks-1.train', dtype='string', delimiter=' ')
# label data
y = X[:,1].astype(int)
y = y[..., np.newaxis]
# input data
X = X[:,2:-1].astype(int)
X.shape[0]
# settings
inputlayer_neurons = int(X.shape[1])
hiddenlayer_neurons = 3
output_neurons = 1
epoch = 100
lr = 0.2
# input
a_0 = tf.placeholder(np.float32, X.shape)
# output
output = tf.placeholder(np.float32, y.shape)
# weight and bias initialization
# hidden layer
w_1 = tf.Variable(tf.truncated_normal([inputlayer_neurons, hiddenlayer_neurons]))
b_1 = tf.Variable(tf.truncated_normal([1, hiddenlayer_neurons]))
# output layer
w_2 = tf.Variable(tf.truncated_normal([hiddenlayer_neurons, output_neurons]))
b_2 = tf.Variable(tf.truncated_normal([1, output_neurons]))
# putting together
z_1 = tf.add(tf.matmul(a_0, w_1), b_1)
a_1 = tf.nn.sigmoid(z_1)
z_2 = tf.add(tf.matmul(a_1, w_2), b_2)
a_2 = tf.nn.sigmoid(z_2)
# error and backprop
diff = tf.subtract(a_2, y)
cost = tf.multiply(diff, diff)
step = tf.train.GradientDescentOptimizer(lr).minimize(cost)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

for i in range(epoch):
    # print('before step {}, y is {}'.format(i, sess.run(output)))
    sess.run(step, feed_dict = {a_0: X, output : y})

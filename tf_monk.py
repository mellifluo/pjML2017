#!/usr/bin/env python

from utils import *

X, y = init()
# settings
inputlayer_neurons = int(X.shape[1])
hiddenlayer_neurons = 5
output_neurons = 1
epoch = 2000
lr = 3
beta = 0.001

graph = tf.Graph()
with graph.as_default():
    # input
    # a0 = tf.constant(X)
    a0 = tf.placeholder(tf.float32, X.shape)
    # output
    # output = tf.constant(y)
    output = tf.placeholder(tf.float32, y.shape)
    # weight and bias initialization
    # hidden layer
    w1 = tf.Variable(tf.random_uniform([inputlayer_neurons, hiddenlayer_neurons], dtype=tf.float32))
    b1 = tf.Variable(tf.random_uniform([1, hiddenlayer_neurons] , dtype=tf.float32))
    # output layer
    w2 = tf.Variable(tf.random_uniform([hiddenlayer_neurons, output_neurons], dtype=tf.float32))
    b2 = tf.Variable(tf.random_uniform([1, output_neurons], dtype=tf.float32))
    # putting together
    a1 = tf.nn.sigmoid(tf.add(tf.matmul(a0, w1), b1))
    a2 = tf.nn.sigmoid(tf.add(tf.matmul(a1, w2), b2))
    # error and backprop
    loss = tf.losses.mean_squared_error(output, a2)
    # regularization
    reg1w = tf.nn.l2_loss(w1) * beta
    reg1b = tf.nn.l2_loss(b1) * beta
    reg2w = tf.nn.l2_loss(w2) * beta
    reg2b = tf.nn.l2_loss(b2) * beta
    totalreg = reg1w + reg1b + reg2w + reg2b
    loss = tf.reduce_mean(loss + totalreg)
    step = tf.train.GradientDescentOptimizer(lr).minimize(loss)
    accuracy = 1 - loss

with tf.Session(graph=graph) as sess:
    sess.run(tf.global_variables_initializer())
    # useful for visualization
    summ_tr = tf.summary.scalar('sse', loss)
    merged_train = tf.summary.merge([summ_tr])
    summ_ts = tf.summary.scalar('accuracy', accuracy)
    merged_test = tf.summary.merge([summ_ts])
    writer_train = tf.summary.FileWriter("output/train/", sess.graph)
    writer_test = tf.summary.FileWriter("output/test/", sess.graph)
    # training
    for i in range(epoch):
        summ, _ = sess.run([merged_train, step], feed_dict={a0: X, output: y})
        writer_train.add_summary(summ, i)
        if i % 100 == 0:
            test = np.loadtxt('./monk/monks-1.test', dtype='string', delimiter=' ')
            testX, testY = prep_data(test)
            testX = norm_data(testX)
            testX = testX[:X.shape[0]]
            testY = testY[:y.shape[0]]
            summ, _ = sess.run([merged_test, step], feed_dict={a0: testX, output: testY})
            writer_test.add_summary(summ, i)
    writer_test.close()
    writer_train.close()

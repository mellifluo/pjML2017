from utils import *

def nn(X, y, hl_u=5, lr=0.1, beta=0.001, fit=False, epoch=1000):
    # settings
    inputlayer_neurons = int(X.shape[1])
    hiddenlayer_neurons = hl_u
    output_neurons = 1
    lr_nn = lr # 0<eta<1
    beta_nn = beta

    graph = tf.Graph()
    with graph.as_default():
        # input
        # a0 = tf.constant(X)
        a0 = tf.placeholder(tf.float32, X.shape, name='input_l')
        # output
        # output = tf.constant(y)
        output = tf.placeholder(tf.float32, y.shape, name='output_l')
        # weight and bias initialization
        # hidden layer
        w1 = tf.Variable(tf.random_uniform([inputlayer_neurons, hiddenlayer_neurons], dtype=tf.float32, name='w_hidden_l'))
        b1 = tf.Variable(tf.random_uniform([1, hiddenlayer_neurons] , dtype=tf.float32, name='wb_hidden_l'))
        # output layer
        w2 = tf.Variable(tf.random_uniform([hiddenlayer_neurons, output_neurons], dtype=tf.float32, name='w_output_l'))
        b2 = tf.Variable(tf.random_uniform([1, output_neurons], dtype=tf.float32, name='wb_output_l'))
        # putting together
        a1 = tf.nn.sigmoid(tf.add(tf.matmul(a0, w1), b1), name='hidden_l')
        a2 = tf.nn.sigmoid(tf.add(tf.matmul(a1, w2), b2), name='output_l')
        # error and backprop
        loss = tf.losses.mean_squared_error(output, a2)
        # regularization
        reg1w = tf.nn.l2_loss(w1) * beta_nn
        reg1b = tf.nn.l2_loss(b1) * beta_nn
        reg2w = tf.nn.l2_loss(w2) * beta_nn
        reg2b = tf.nn.l2_loss(b2) * beta_nn
        with tf.name_scope('reg'):
            totalreg = reg1w + reg1b + reg2w + reg2b
        loss = tf.reduce_mean(loss + totalreg, name='loss')
        with tf.name_scope('step'):
            step = tf.train.GradientDescentOptimizer(lr_nn).minimize(loss)
        with tf.name_scope('accuracy'):
            accuracy = 1 - loss
        # useful for visualization
        summ_tr = tf.summary.scalar('sse', loss)
        merged_train = tf.summary.merge([summ_tr], name='mtr')
        summ_ts = tf.summary.scalar('accuracy', accuracy)
        merged_test = tf.summary.merge([summ_ts], name='mte')
    if fit:
        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            writer_train = tf.summary.FileWriter("output/train/lr" + str(lr), sess.graph)
            writer_test = tf.summary.FileWriter("output/test/lr" + str(lr), sess.graph)
            # training
            for i in range(epoch):
                summ, _ = sess.run([merged_train, step], feed_dict={a0: X, output: y})
                writer_train.add_summary(summ, i)
                if i % 100 == 0:
                    testX, testY = test_data(X.shape, y.shape)
                    summ, _ = sess.run([merged_test, step], feed_dict={a0: testX, output: testY})
                    writer_test.add_summary(summ, i)
            writer_test.close()
            writer_train.close()
    else:
        return graph

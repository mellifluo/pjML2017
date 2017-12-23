from utils import *

def nn(X, y, hl_u=5, lr=0.1, beta=0.001, fit=False, epoch=100, tanh=True):
    """
    This function creates a model depending on arguments:
        X: set of records without labels
        y: set of labels
        hl_u: number of hidden layer units
        lr: learning rate (eta)
        beta: regularization term
        fit: boolean value concerning fitting the model or not
        epoch: how many epochs in the training phase
        tanh: using or not tanh as activation function (instead of sigmoid)

    Returns the graph of the model or writes the tensorboard's files in the
    folder './output'
    """
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
        a0 = tf.placeholder(tf.float32, name='input_l')
        # output
        # output = tf.constant(y)
        output = tf.placeholder(tf.float32, name='output_l')
        # weight and bias initialization
        # hidden layer
        w1 = tf.Variable(tf.random_normal([inputlayer_neurons, hiddenlayer_neurons], dtype=tf.float32, name='w_hidden_l'))
        b1 = tf.Variable(tf.random_normal([hiddenlayer_neurons] , dtype=tf.float32, name='wb_hidden_l'))
        # output layer
        w2 = tf.Variable(tf.random_normal([hiddenlayer_neurons, output_neurons], dtype=tf.float32, name='w_output_l'))
        b2 = tf.Variable(tf.random_normal([output_neurons], dtype=tf.float32, name='wb_output_l'))
        # putting together
        if tanh:
            a1 = tf.nn.tanh(tf.nn.bias_add(tf.matmul(a0, w1), b1), name='hidden_l')
            a2 = tf.nn.tanh(tf.nn.bias_add(tf.matmul(a1, w2), b2), name='output_l')
        else:
            a1 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(a0, w1), b1), name='hidden_l')
            a2 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(a1, w2), b2), name='output_l')
        # error and backprop
        loss = tf.losses.mean_squared_error(output, a2)
        # regularization
        reg1w = tf.nn.l2_loss(w1) * beta_nn
        reg1b = tf.nn.l2_loss(b1) * beta_nn
        reg2w = tf.nn.l2_loss(w2) * beta_nn
        reg2b = tf.nn.l2_loss(b2) * beta_nn
        with tf.name_scope('reg'):
            totalreg = reg1w + reg1b + reg2w + reg2b
        loss = tf.add(loss, totalreg, name='loss')
        with tf.name_scope('step'):
            step = tf.train.GradientDescentOptimizer(lr_nn).minimize(loss)
        with tf.name_scope('accuracy_ev'):
            if tanh:
                corrects_plus1 = tf.add(a2,1)
                corrects_int = tf.cast(corrects_plus1, tf.int64)
            else:
                corrects_int = tf.round(a2)
                corrects_int = tf.cast(corrects_int, tf.int64)
            output_int = tf.cast(output, tf.int64)
            corrects = tf.equal(corrects_int, output_int)
            accuracy = tf.reduce_mean(tf.cast(corrects, 'float32'))
        # useful for visualization
        summ_tr = tf.summary.scalar('sse', loss)
        merged_train = tf.summary.merge([summ_tr], name='mtr')
        summ_ts = tf.summary.scalar('accuracy', accuracy)
        merged_test = tf.summary.merge([summ_ts], name='mte')
    if fit:
        with tf.Session(graph=graph) as sess:
            sess.run(tf.global_variables_initializer())
            str_tr = "output/train/lr" + str(lr) + "b" + str(beta) + "hl" + str(hl_u) + "ep" + str(epoch)
            str_te = "output/test/lr" + str(lr) + "b" + str(beta) + "hl" + str(hl_u) + "ep" + str(epoch)
            writer_train = tf.summary.FileWriter(str_tr, sess.graph)
            writer_test = tf.summary.FileWriter(str_te, sess.graph)
            # training
            for i in range(epoch):
                _ = sess.run(step, feed_dict={a0: X, output: y})
                if i % 10 == 0:
                    summ1 = sess.run(merged_train, feed_dict={a0: X, output: y})
                    writer_train.add_summary(summ1, i)
                    testX, testY = test_data(X.shape, y.shape)
                    corr, summ2 = sess.run([accuracy, merged_test], feed_dict={a0: testX, output: testY})
                    writer_test.add_summary(summ2, i)
            writer_test.close()
            writer_train.close()
    else:
        return graph

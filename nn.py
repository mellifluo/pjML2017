from utils import *
from validation import *

def nn(X, hl_u=5, lr=0.1, mom=0.9, alpha=0, fit=True, epoch=100, tanh=True, addstr='', cv=3):
    """
    This function creates a model depending on arguments:
        X: dataset
        hl_u: number of hidden layer units
        lr: learning rate (eta)
        alpha: regularization term
        fit: boolean value concerning fitting the model or not
        epoch: how many epochs in the training phase
        tanh: using or not tanh as activation function (instead of sigmoid)
        addstr: string to add in tensorboard files
        cv: cross-validation parameter
    Returns the graph of the model or writes the tensorboard's files in the
    folder './output'
    """
    if cv:
        X = cross_validation(X, cv)
        testX = X[2,:,:]
        testX, testY = split_classes(testX)
        valX = X[1,:,:]
        valX, valY = split_classes(valX)
        X = X[0,:,:]
        X, y = split_classes(X)
        if tanh:
            testY = np.where(testY == 0, -1, testY)
            valY = np.where(valY == 0, -1, valY)
            y = np.where(y == 0, -1, y)
    else:
        X, y = init()
        valX, valY = init()
        testX, testY = test_data()
        if tanh:
            testY = np.where(testY == 0, -1, testY)
            valY = np.where(valY == 0, -1, valY)
            y = np.where(y == 0, -1, y)
    # settings
    inputlayer_neurons = int(X.shape[1])
    hiddenlayer_neurons = hl_u
    output_neurons = 1
    lr_nn = lr # 0<eta<1
    alpha_nn = alpha
    # w1_np = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))
    # b1_np = np.random.uniform(size=(hiddenlayer_neurons))
    # w2_np = np.random.uniform(size=(hiddenlayer_neurons, output_neurons))
    # b2_np = np.random.uniform(size=(output_neurons))

    graph = tf.Graph()
    with graph.as_default():
        # input
        a0 = tf.placeholder(tf.float32, name='input_l')
        # output
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
        reg1w = tf.nn.l2_loss(w1) * alpha_nn
        reg1b = tf.nn.l2_loss(b1) * alpha_nn
        reg2w = tf.nn.l2_loss(w2) * alpha_nn
        reg2b = tf.nn.l2_loss(b2) * alpha_nn
        with tf.name_scope('reg'):
            totalreg = reg1w + reg1b + reg2w + reg2b
        loss2 = tf.add(loss, totalreg, name='loss')
        with tf.name_scope('step'):
            step = tf.train.MomentumOptimizer(lr_nn, momentum=mom).minimize(loss2)
        with tf.name_scope('accuracy_ev'):
            if tanh:
                # corrects_plus1 = tf.add(a2,0.999999)
                # corrects_int = tf.cast(corrects_plus1, tf.int64)
                # cond = tf.greater(a2,tf.zeros_like(a2))
                # corrects_int = tf.where(cond, tf.ones_like(a2), tf.zeros_like(a2))
                corrects_int = tf.round(a2)
            else:
                corrects_int = tf.round(a2)
            corrects = tf.equal(corrects_int, output)
            accuracy = tf.reduce_mean(tf.cast(corrects, 'float32'))
        # useful for visualization
        summ_tr = tf.summary.scalar('sse', loss)
        summ_tr_acc = tf.summary.scalar('acc_train', accuracy)
        merged_train = tf.summary.merge([summ_tr, summ_tr_acc], name='mtr')
        summ_vl = tf.summary.scalar('error_val', loss)
        merged_val = tf.summary.merge([summ_vl], name='mvl')
        summ_ts = tf.summary.scalar('accuracy', accuracy)
        merged_test = tf.summary.merge([summ_ts], name='mte')
    if fit:
        if not cv: cv = 1
        for jj in range(cv):
            with tf.Session(graph=graph) as sess:
                sess.run(tf.global_variables_initializer())
                addstr = "val" + str(jj)
                # logs for tensorboard
                str_tr = "output/train/lr" + str(lr) + "b" + str(alpha) + "hl" + str(hl_u) + "ep" + str(epoch) + str(addstr)
                str_vl = "output/val/lr" + str(lr) + "b" + str(alpha) + "hl" + str(hl_u)  + "ep" + str(epoch) + str(addstr)
                str_te = "output/test/lr" + str(lr) + "b" + str(alpha) + "hl" + str(hl_u)  + "ep" + str(epoch) + str(addstr)
                writer_train = tf.summary.FileWriter(str_tr, sess.graph)
                writer_vl = tf.summary.FileWriter(str_vl, sess.graph)
                writer_test = tf.summary.FileWriter(str_te, sess.graph)
                # training
                for i in range(epoch):
                    _ = sess.run(step, feed_dict={a0: X, output: y})
                    # validation & test
                    if i % 10 == 0:
                        acc, summ1 = sess.run([a2, merged_train], feed_dict={a0: X, output: y})
                        writer_train.add_summary(summ1, i)
                        corr, summ2 = sess.run([loss, merged_val], feed_dict={a0: valX, output: valY})
                        corr, summ3 = sess.run([accuracy, merged_test], feed_dict={a0: testX, output: testY})
                        writer_vl.add_summary(summ2, i)
                        writer_test.add_summary(summ3, i)
                writer_test.close()
                writer_vl.close()
                writer_train.close()
                sess.close()
            if jj == 0:
                three_switch(X,valX,testX)
                three_switch(y,valY,testY)
            else:
                three_switch(testX,X,valX)
                three_switch(testY,y,valY)
    else:
        return graph

X, y = init()
nn(X, lr=0.5, cv=None, epoch=100, tanh=True, hl_u=5)

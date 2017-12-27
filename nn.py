from utils import *
from validation import *

def nn(df,lab, d, hl_u=5, lr=0.1, mom=0.9, alpha=0, fit=True, epoch=100, tanh=True, addstr='', cv=3):
    """
    This function creates a model depending on arguments:
        df: feature set
        lab: label set
        d: dataset (monk1,2,3,4,cup)
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
        idxs = cross_validation2(cv)
        testX, testY = test_data(d)
        if tanh:
            testY = np.where(testY == 0, -1, testY)
            lab = np.where(lab == 0, -1, lab)
    else:
        X = df
        y = lab
        valX, valY = init(d)
        testX, testY = test_data(d)
        if tanh:
            testY = np.where(testY == 0, -1, testY)
            valY = np.where(valY == 0, -1, valY)
            y = np.where(y == 0, -1, y)
    # settings
    inputlayer_neurons = int(df.shape[1])
    hiddenlayer_neurons = hl_u
    output_neurons = 1
    lr_nn = lr # 0<eta<1
    alpha_nn = alpha

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
        with tf.name_scope('loss'):
            loss2 = loss + totalreg
        with tf.name_scope('step'):
            step = tf.train.MomentumOptimizer(lr_nn, momentum=mom).minimize(loss2)
        with tf.name_scope('accuracy_ev'):
            corrects_int = tf.round(a2)
            corrects = tf.equal(corrects_int, output)
            accuracy = tf.reduce_mean(tf.cast(corrects, 'float32'))
        # useful for visualization
        summ_tr = tf.summary.scalar('sse', loss)
        summ_tr_acc = tf.summary.scalar('acc_train', accuracy)
        merged_train = tf.summary.merge([summ_tr, summ_tr_acc], name='mtr')
        summ_vl = tf.summary.scalar('error_val', loss)
        summ_vl_acc = tf.summary.scalar('acc_val', accuracy)
        merged_val = tf.summary.merge([summ_vl, summ_vl_acc], name='mvl')
        summ_ts = tf.summary.scalar('accuracy', accuracy)
        merged_test = tf.summary.merge([summ_ts], name='mte')
    if fit:
        if not cv: cv = 1
        # qui c'è la parte nuova
        #capito
        for jj in range(cv):
            if cv != 1:
                maskX = np.ones(len(df), dtype=bool)
                maskY = np.ones(len(lab), dtype=bool)
                # fare stessa cosa per y
                if idxs[jj]:
                    valX = df[idxs[jj]+1:idxs[jj+1]]
                    maskX[idxs[jj]+1:idxs[jj+1]] = False
                    valY = lab[idxs[jj]+1:idxs[jj+1]]
                    maskY[idxs[jj]+1:idxs[jj+1]] = False
                else:
                    valX = df[idxs[jj]:idxs[jj+1]]
                    maskX[idxs[jj]:idxs[jj+1]] = False
                    valY = lab[idxs[jj]:idxs[jj+1]]
                    maskY[idxs[jj]:idxs[jj+1]] = False
                X = df[maskX, :]
                y = lab[maskY, :]
                pX = np.zeros(shape=(cv-1,valX.shape[0],valX.shape[1]))
                pY = np.zeros(shape=(cv-1,valY.shape[0],valY.shape[1]))
                for subparts in range(cv-1):
                    pX[subparts,:,:] = X[:len(valX),:]
                    X = X[len(valX):,:]
                    pY[subparts,:,:] = y[:len(valY),:]
                    y = y[len(valY):,:]
            with tf.Session(graph=graph) as sess:
                sess.run(tf.global_variables_initializer())
                addstr = "val" + str(jj)
                if cv != 1: print addstr, idxs[jj], idxs[jj+1]
                # logs for tensorboard
                str_tr = "output/train/lr" + str(lr) + "b" + str(alpha) + "hl" + str(hl_u) + "ep" + str(epoch) + str(addstr)
                str_vl = "output/val/lr" + str(lr) + "b" + str(alpha) + "hl" + str(hl_u)  + "ep" + str(epoch) + str(addstr)
                str_te = "output/test/lr" + str(lr) + "b" + str(alpha) + "hl" + str(hl_u)  + "ep" + str(epoch) + str(addstr)
                writer_train = tf.summary.FileWriter(str_tr, sess.graph)
                writer_vl = tf.summary.FileWriter(str_vl, sess.graph)
                writer_test = tf.summary.FileWriter(str_te, sess.graph)
                # training
                for i in range(epoch):
                    if cv != 1:
                        for subparts in range(cv-1):
                            _ = sess.run(step, feed_dict={a0: pX[subparts], output: pY[subparts]})
                        summ1 = sess.run(merged_train, feed_dict={a0: pX[subparts], output: pY[subparts]})
                        writer_train.add_summary(summ1, i)
                    else:
                        _, summ1 = sess.run([step, merged_train], feed_dict={a0: X, output: y})
                        writer_train.add_summary(summ1, i)
                    # validation & test
                    if i % 10 == 0:
                        summ2 = sess.run(merged_val, feed_dict={a0: valX, output: valY})
                        summ3 = sess.run(merged_test, feed_dict={a0: testX, output: testY})
                        writer_vl.add_summary(summ2, i)
                        writer_test.add_summary(summ3, i)
                writer_test.close()
                writer_vl.close()
                writer_train.close()
                sess.close()
    else:
        return graph
d=1
X, y = init(d)
nn(X, y, d, lr=0.5, epoch=500, tanh=True, hl_u=10, cv=None)

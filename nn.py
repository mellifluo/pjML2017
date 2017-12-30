from utils import *
from validation import *

def nn(df,lab, d, bs=None, hl_u=5, lr=0.1, mom=0.9, alpha=0, epoch=100, tanh=True, cv=3):
    """
    This function creates a model depending on arguments:
        df: feature set
        lab: label set
        d: dataset (monk1,2,3,4,cup)
        bs: batch size (if None, full batch)
        hl_u: number of hidden layer units
        lr: learning rate (eta)
        alpha: regularization term
        fit: boolean value concerning fitting the model or not
        epoch: how many epochs in the training phase
        tanh: using or not tanh as activation function (instead of sigmoid)
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
    cv_all = []
    graph = tf.Graph()
    with graph.as_default():
        # input
        a0 = tf.placeholder(tf.float32, name='input_l')
        # output
        output = tf.placeholder(tf.float32, name='output_l')
        # weight and bias initialization
        # hidden layer
        w1 = tf.Variable(tf.random_normal([inputlayer_neurons, hiddenlayer_neurons], stddev=1, seed=1, dtype=tf.float32, name='w_hidden_l'))
        b1 = tf.Variable(tf.random_normal([hiddenlayer_neurons], dtype=tf.float32, name='wb_hidden_l'))
        # output layer
        w2 = tf.Variable(tf.random_normal([hiddenlayer_neurons, output_neurons], stddev=1, seed=2, dtype=tf.float32, name='w_output_l'))
        b2 = tf.Variable(tf.random_normal([output_neurons], dtype=tf.float32, name='wb_output_l'))
        # w1 = (w1*2)/inputlayer_neurons
        # w2 = (w2*2)/inputlayer_neurons
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
            step = tf.train.MomentumOptimizer(lr_nn, momentum=mom, use_nesterov=True).minimize(loss2)
        with tf.name_scope('accuracy_ev'):
            if tanh:
                cond = tf.greater_equal(a2,tf.zeros_like(a2))
                minus_one = tf.zeros_like(a2) - 1
                corrects_int = tf.where(cond, tf.ones_like(a2), minus_one)
            else:
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
        acc_val = 0
        acc_test = 0
        cvscores = []
        if not cv: cv = 1
        for jj in range(cv):
            if cv != 1:
                maskX = np.ones(len(df), dtype=bool)
                maskY = np.ones(len(lab), dtype=bool)
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
            with tf.Session(graph=graph) as sess:
                sess.run(tf.global_variables_initializer())
                print "lr="+str(lr), "hl="+str(hl_u)
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
                    if bs:
                        idx = np.random.randint(X.shape[0], size=bs)
                        xp = np.matrix(X[idx,:])
                        yp = np.matrix(y[idx,:])
                        for steps in range(int(X.shape[0]/bs)):
                            _, summ1 = sess.run([step, merged_train], feed_dict={a0: xp, output: yp})
                        # writer_train.add_summary(summ1, i)
                    else:
                        _, summ1 = sess.run([step, merged_train], feed_dict={a0: X, output: y})
                        writer_train.add_summary(summ1, i)
                    # validation & test
                    acc_val, summ2 = sess.run([accuracy,merged_val], feed_dict={a0: valX, output: valY})
                    acc_test, summ3 = sess.run([accuracy,merged_test], feed_dict={a0: testX, output: testY})
                    # writer_vl.add_summary(summ2, i)
                    # writer_test.add_summary(summ3, i)
                acc_val = acc_val * 100
                acc_test = acc_test * 100
                cvscores.append(acc_val)
                print "acc_val: %.2f%%" % acc_val
                print "acc_test: %.2f%%" % acc_test
                print " "
                writer_test.close()
                writer_vl.close()
                writer_train.close()
                sess.close()
        print "CV", "lr="+str(lr), "hl="+str(hl_u)
        print "%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores))
        print " "
        return [np.mean(cvscores), np.std(cvscores), lr, hl_u]

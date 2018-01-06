from utils import *

def nn(d, bs=32, hl_u=5, lr=0.1, mom=0.9, alpha=0, epoch=100, tanh=True, nest=True, cv=3, dbcv=False):
    """
    This function creates a model depending on arguments:
        d: dataset (monk1,2,3 or 4 cup)
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
    df, lab = init(d, shuffle=True)
    if d == 4:
        if not dbcv:
            cvpercent = int(df.shape[0]*(.3))
            testXcv = df[:cvpercent,:]
            testYcv = lab[:cvpercent,:]
            df = df[cvpercent:,:]
            lab = lab[cvpercent:,:]
        testX, testY = test_data(d)
        tanh = False
    else: testX, testY = test_data(d)
    if cv:
        idxs = cross_validation(len(df), cv, d)
        if tanh and d != 4:
            testY = np.where(testY == 0, -1, testY)
            lab = np.where(lab == 0, -1, lab)
    else:
        X, y = df, lab
        valX, valY = init(d)
        if tanh and d != 4:
            testY = np.where(testY == 0, -1, testY)
            valY = np.where(valY == 0, -1, valY)
            y = np.where(y == 0, -1, y)
    # settings
    inputlayer_neurons = int(df.shape[1])
    hiddenlayer_neurons = hl_u
    if d == 4: output_neurons = 2
    else: output_neurons = 1
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
        fanin1 = np.sqrt((6.0)/(inputlayer_neurons + hiddenlayer_neurons))
        w1 = tf.Variable(tf.random_uniform([inputlayer_neurons, hiddenlayer_neurons], minval=-fanin1, maxval=fanin1, dtype=tf.float32, name='w_hidden_l'))
        b1 = tf.Variable(tf.zeros([hiddenlayer_neurons], dtype=tf.float32, name='wb_hidden_l'))
        # output layer
        fanin2 = np.sqrt((6.0)/(output_neurons + hiddenlayer_neurons))
        w2 = tf.Variable(tf.random_uniform([hiddenlayer_neurons, output_neurons], minval=-fanin2, maxval=fanin2, dtype=tf.float32, name='w_output_l'))
        b2 = tf.Variable(tf.zeros([output_neurons], dtype=tf.float32, name='wb_output_l'))
        # putting together
        if tanh and d != 4:
            a1 = tf.nn.tanh(tf.nn.bias_add(tf.matmul(a0, w1), b1), name='hidden_l')
            a2 = tf.nn.tanh(tf.nn.bias_add(tf.matmul(a1, w2), b2), name='output_l')
        elif d == 4:
            a1 = tf.nn.tanh(tf.nn.bias_add(tf.matmul(a0, w1), b1, name='hidden_l'))
            a2 = tf.nn.bias_add(tf.matmul(a1, w2), b2, name='output_l')
        else:
            a1 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(a0, w1), b1), name='hidden_l')
            a2 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(a1, w2), b2), name='output_l')
        # error and backprop
        if d == 4: loss = mean_euc_dist(output,a2)
        else: loss = tf.losses.mean_squared_error(output, a2)
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
            step = tf.train.MomentumOptimizer(lr_nn, momentum=mom, use_nesterov=nest).minimize(loss2)
        with tf.name_scope('accuracy_ev'):
            if tanh:
                cond = tf.greater_equal(a2,tf.zeros_like(a2))
                minus_one = tf.zeros_like(a2) - 1
                corrects_int = tf.where(cond, tf.ones_like(a2), minus_one)
            elif d != 4:
                corrects_int = tf.round(a2)
            else:
                corrects_int = a2
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
                mask = np.ones(len(df), dtype=bool)
                if idxs[jj]:
                    valX = df[idxs[jj]+1:idxs[jj+1]]
                    valY = lab[idxs[jj]+1:idxs[jj+1]]
                    mask[idxs[jj]+1:idxs[jj+1]] = False
                else:
                    valX = df[idxs[jj]:idxs[jj+1]]
                    valY = lab[idxs[jj]:idxs[jj+1]]
                    mask[idxs[jj]:idxs[jj+1]] = False
                X = df[mask, :]
                y = lab[mask, :]
                if d == 4 and dbcv:
                    mask = np.ones(len(X), dtype=bool)
                    if idxs[jj]:
                        testX = X[idxs[jj]+1:idxs[jj+1]]
                        testY = y[idxs[jj]+1:idxs[jj+1]]
                        mask[idxs[jj]+1:idxs[jj+1]] = False
                    else:
                        testXcv = X[idxs[jj]:idxs[jj+1]]
                        testYcv = y[idxs[jj]:idxs[jj+1]]
                        mask[idxs[jj]:idxs[jj+1]] = False
                    X = X[mask, :]
                    y = y[mask, :]
            with tf.Session(graph=graph) as sess:
                sess.run(tf.global_variables_initializer())
                print "lr="+str(lr), "hl="+str(hl_u)
                addstr = "val" + str(jj)
                if cv != 1: print addstr, idxs[jj], idxs[jj+1]
                if d == 4 and dbcv:
                    if idxs[jj] > idxs[-1]/2: print "ts" + str(jj), idxs[jj-1], idxs[jj]
                    else: print "ts" + str(jj), idxs[jj+1], idxs[jj+2]
                # logs for tensorboard
                str_tr = "output/train/lr" + str(lr) + "a" + str(alpha) + "hl" + str(hl_u) + "ep" + str(epoch) + str(addstr)
                str_vl = "output/val/lr" + str(lr) + "a" + str(alpha) + "hl" + str(hl_u)  + "ep" + str(epoch) + str(addstr)
                str_te = "output/test/lr" + str(lr) + "a" + str(alpha) + "hl" + str(hl_u)  + "ep" + str(epoch) + str(addstr)
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
                    else:
                        _, summ1 = sess.run([step, merged_train], feed_dict={a0: X, output: y})
                    if cv == 1: writer_train.add_summary(summ1, i)
                    # validation & test
                    if d == 4:
                        res, acc_val, summ2 = sess.run([a2,loss,merged_val], feed_dict={a0: valX, output: valY})
                        acc_test, summ3 = sess.run([loss,merged_test], feed_dict={a0: testXcv, output: testYcv})
                    else:
                        lval, acc_val, summ2 = sess.run([loss, accuracy,merged_val], feed_dict={a0: valX, output: valY})
                        lts, acc_test, summ3 = sess.run([loss, accuracy,merged_test], feed_dict={a0: testX, output: testY})
                        if cv == 1: writer_test.add_summary(summ3, i)
                        acc_val = acc_val * 100
                        acc_test = acc_test * 100
                    if cv == 1: writer_vl.add_summary(summ2, i)
                if d== 4:
                    print "loss_val: %.2f" % acc_val
                    print "l_test: %.2f" % acc_test
                else:
                    print "acc_val: %.2f%%" % acc_val
                    print "acc_test: %.2f%%" % acc_test
                    print "l_val: %.10f" % lval
                    print "l_test: %.10f" % lts
                print " "
                cvscores.append(acc_test)
                writer_test.close()
                writer_vl.close()
                writer_train.close()
                sess.close()
        print "CV", "lr="+str(lr), "hl="+str(hl_u)
        if d == 4:
            print "%.2f (+/- %.2f)" % (np.mean(cvscores), np.std(cvscores))
            np.savetxt("tf_CUPresults.csv", res, delimiter=',')
        else: print "%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores))
        print " "
        return [np.mean(cvscores), np.std(cvscores), lr, hl_u]

from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers
import keras
from utils import *
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import seaborn as sns

# define MEE
from keras import backend as K
def mean_euc_dist(y_true, y_pred):
    return K.mean(K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True)))

# define model
def func_model(f,nodes,lr,alpha,out_u):
    # create the structure of the model
    inputs = Input(shape=(X.shape[1],))
    x = Dense(nodes,activation=f)(inputs)
    predictions = Dense(out_u)(x)
    model = Model(inputs=inputs, outputs=predictions)
    # set gradient descent
    sgd = optimizers.SGD(lr=lr, decay=alpha, momentum=0.9, nesterov=True)
    # differences on loss (depending on the task)
    if out_u == 1:
        model.compile(optimizer=sgd,
                      loss='mean_squared_error',
                      metrics=['accuracy'])
    elif out_u == 2:
        model.compile(optimizer=sgd,
                      loss=mean_euc_dist,
                      metrics=['accuracy'])
    return model

def clas_nn():
    fig = plt.figure(figsize=(20, 20))
    fig_dims = (3, 3)
    sns.set()
    assx=0
    for hl in hls:
        assy=0
        for lr in lrs:
            model = func_model('tanh',hl,lr,0.000001,1)
            kfold = KFold(n_splits=5)
            cvscores = []
            # cross-validation
            for train, test in kfold.split(X, y):
            	# Fit the model
            	model.fit(X[train], y[train], epochs=100, verbose=0, shuffle=False)
            	# evaluate the model
            	scores = model.evaluate(X[test], y[test], verbose=0)
            	cvscores.append(scores[1] * 100)
            print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
            cvs.append([np.mean(cvscores), np.std(cvscores), lr, hl])
            # re-train with hold out
            model = func_model('tanh',hl,lr,0,1)
            history = model.fit(X,y, epochs=100, verbose=0, validation_data=(testX,testy), shuffle=False)
            print 'MSE:' ,history.history['loss'][99]
            print 'MSE test:' ,history.history['val_loss'][99]
            print 'acc test',history.history['val_acc'][99]
            plt.subplot2grid(fig_dims, (assx, assy))
            plt.xlabel('Epochs')
            plt.title('Accuracy')
            plt.plot(history.history['acc'],label="acc_train")
            plt.plot(history.history['val_acc'], label='acc_test',linestyle='--')
            plt.legend(loc='best')
            plt.subplot2grid(fig_dims, (assx, assy+1))
            plt.xlabel('Epochs')
            plt.title('Loss')
            plt.plot(history.history['loss'],label="loss_train")
            plt.plot(history.history['val_loss'], label='loss_test',linestyle='--')
            plt.legend(loc='best')
            assy=assy+1
        assx=assx+1
    plt.show()
    fig.savefig('kerasGrid.png')
    # (sort of) model selection (a really greedy choice, not used for the project)
    m = max([b[0] for b in cvs])
    best = next((x for x in cvs if m == x[0]), None)
    print "-------------"
    print "Best with lr=%.2f and %d hidden layer units:" % (best[2], best[3])
    print "%.1f%% (+/- %.1f%%)" % (best[0], best[1])

#############################################################################
def reg_nn():
    fig = plt.figure(figsize=(8, 8))
    fig_dims = (3, 3)
    sns.set()
    assx=0
    for hl in hls:
        assy=0
        for lr in lrs:
            model = func_model('tanh',hl,lr,1e-3,out_u=2)
            kfold = KFold(n_splits=3)
            cvscores = []
            # set aside test (30% of dataset)
            cvpercent = int(X.shape[0]*(.3))
            testXval = X[:cvpercent,:]
            testyval = y[:cvpercent,:]
            Xtr = X[cvpercent:,:]
            ytr = y[cvpercent:,:]
            # cross-validation
            for train, test in kfold.split(X, y):
            	# Fit the model
            	model.fit(X[train], y[train], epochs=100, verbose=0, shuffle=False)
            	# evaluate the model
            	scores = model.evaluate(X[test], y[test], verbose=0)
            	print("%s: %.2f" % (model.metrics_names[0], scores[0]))
            	cvscores.append(scores[0])
            print("%.2f (+/- %.2f)" % (np.mean(cvscores), np.std(cvscores)))
            cvs.append([np.mean(cvscores), np.std(cvscores), lr, hl])
            # re-train with hold out
            model = func_model('tanh',hl,lr,1e-3,2)
            history = model.fit(Xtr,ytr, epochs=100, verbose=0, validation_data=(testXval,testyval), shuffle=False)
            print 'MSE:' ,history.history['loss'][99]
            print 'MSE test:' ,history.history['val_loss'][99]
            plt.subplot2grid(fig_dims, (assx, assy))
            plt.xlabel('Epochs')
            plt.tight_layout()
            plt.title('lr='+ str(lr)+" U="+str(hl))
            plt.plot(history.history['loss'],label="MEE_TR")
            plt.plot(history.history['val_loss'], label='MEE_VL',linestyle='--')
            plt.legend(loc='best')
            assy=assy+1
        assx=assx+1
    plt.show()
    fig.savefig('./gridReg.png')
    # (sort of) model selection (a really greedy choice, not used for the project)
    m = min([b[0] for b in cvs])
    best = next((x for x in cvs if m == x[0]), None)
    print "-------------"
    print "Best with lr=%.2f and %d hidden layer units:" % (best[2], best[3])
    print "%.2f (+/- %.2f)" % (best[0], best[1])
    # hold out
    model = func_model('tanh', best[3], best[2],1e-3, out_u=2)
    model.fit(X, y, epochs=100, verbose=0, shuffle=False)
    res = model.predict(testX)
    # save csv
    import pandas as pd
    df = pd.DataFrame(res)
    df.index += 1
    df.to_csv("k_CUPresults.csv", header=None)

# settings
d=4
X, y = init(d, shuffle=True)
testX,testy=test_data(d)
lrs = [0.01]
hls = [10]
cvs = []
# if using MONK (with tanh)
if d != 4:
    y = np.where(y == 0, -1, y)
    testy = np.where(testy == 0, -1, testy)

# command to run regression (so, CUP benchmark)
# reg_nn()
###############
# command to run classification (so, MONKs benchmark)
# clas_nn()

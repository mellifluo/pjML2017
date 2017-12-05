# utils to dataset
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, shutil


def norm_data(X):
    for j in range(X.shape[1]):
        newj = int(X.shape[1] + len(np.unique(X[:,j])) - 1)
        newcols = np.zeros((X.shape[0], abs(newj-X.shape[1])))
        X = np.append(X, newcols, axis=1)
        for ji in np.unique(X[:,j])[:-1]:
            lastcol = list(np.unique(X[:,j])).index(int(ji)) + 1
            X[:,-lastcol] = np.array([int(jj == ji) for jj in X[:,j]])
        X[:,j] = np.array([int(jj == max((X[:,j]))) for jj in X[:,j]])
    return X

def prep_data(X):
    # label data
    y = X[:,1].astype('float32')
    y = y[..., np.newaxis]
    # input data
    X = X[:,2:-1].astype('float32')
    return X, y

def test_data(shapeX, shapeY):
    test = np.loadtxt('./monk/monks-1.test', dtype='string', delimiter=' ')
    testX, testY = prep_data(test)
    testX = norm_data(testX)
    testX = testX[:shapeX[0]]
    testY = testY[:shapeY[0]]
    return testX, testY

def init():
    if os.path.exists('./output'):
        shutil.rmtree('./output')
    # load dataset
    X = np.loadtxt('./monk/monks-1.train', dtype='string', delimiter=' ')
    X, y = prep_data(X)
    X = norm_data(X).astype('float32')
    X.shape
    return X, y

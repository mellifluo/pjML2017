# utils to dataset
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

def prep_data(X):
    for j in range(X.shape[1]):
        newj = int(X.shape[1] + max((X[:,j])) - 1)
        newcols = np.zeros((X.shape[0], abs(newj-X.shape[1])))
        X = np.append(X, newcols, axis=1)
        for ji in range(X.shape[1], newj):
            X[:,-(newj-ji)] = np.array([int(jj == (newj-ji)) for jj in X[:,j]])
        X[:,j] = np.array([int(jj == max((X[:,j]))) for jj in X[:,j]])
    return X

def init():
    files = glob.glob('output/*')
    for f in files:
        os.remove(f)
    tf.reset_default_graph()
    # load dataset
    X = np.loadtxt('./monk/monks-1.train', dtype='string', delimiter=' ')
    # label data
    y = X[:,1].astype('float32')
    y = y[..., np.newaxis]
    # input data
    X = X[:,2:-1].astype('float32')
    X = prep_data(X).astype('float32')
    X.shape
    return X, y

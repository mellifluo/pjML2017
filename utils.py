# utils to dataset
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, shutil
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def norm_data(X):
    """
    returns normalized data on attribute's values 0 or 1
    """
    for j in range(X.shape[1]):
        newj = int(X.shape[1] + len(np.unique(X[:,j])) - 1)
        newcols = np.zeros((X.shape[0], abs(newj-X.shape[1])))
        X = np.append(X, newcols, axis=1)
        for ji in np.unique(X[:,j])[:-1]:
            lastcol = list(np.unique(X[:,j])).index(int(ji)) + 1
            X[:,-lastcol] = np.array([int(jj == ji) for jj in X[:,j]])
        X[:,j] = np.array([int(jj == max((X[:,j]))) for jj in X[:,j]])
    return X

def clean_data(X):
    """
    deletes the first and the last column of monk (that are useless)
    """
    return X[:,1:len(X[1])-1].astype('float32')

def split_classes(X):
    """
    returns the feature set and its label set
    """
    y = X[:,0].astype('float32')
    y = y[..., np.newaxis]
    X = X[:,1:]
    return X, y

def prep_data(X):
    """
    split label set and attribute set and returns them
    """
    # input data
    X = clean_data(X)
    # label data
    X, y = split_classes(X)
    return X, y

def test_data(d=4):
    """
    returns a test set (split on attributes and labels)
    """
    if d == 4:
        testX = np.loadtxt('./cup/ML-CUP17-TS.csv', dtype='float32', delimiter=',')
        testX = testX[:,1:]
        testY = np.ones(shape=(testX.shape[0],2))
        return testX, testY
    elif d==1 or d==2 or d==3:
        test = np.loadtxt('./monk/monks-'+str(d)+'.test', dtype='string', delimiter=' ')
        testX, testY = prep_data(test)
        testX = norm_data(testX)
        return testX.astype('float32'), testY.astype('float32')
    else:
        print('Error: d must be 1,2,3 or 4.')
        return None

def init(d=4, shuffle=True):
    """
    prepare the environment and the data
    returns the normalized data
    """
    if os.path.exists('./output'):
        shutil.rmtree('./output')
    # load dataset
    if d == 4:
        X = np.loadtxt('./cup/ML-CUP17-TR.csv', dtype='float32', delimiter=',')
        X = X[:,1:]
        y = X[:,-2:X.shape[1]]
        X = X[:,:-2]
        if shuffle:
            np.random.shuffle(X)
    elif d==1 or d==2 or d==3:
        X = np.loadtxt('./monk/monks-'+str(d)+'.train', dtype='string', delimiter=' ')
        if shuffle:
            np.random.shuffle(X)
        X, y = prep_data(X)
        X = norm_data(X).astype('float32')
    else:
        print('Error: d must be 1,2,3 or 4.')
        return None
    return X.astype('float32'), y.astype('float32')

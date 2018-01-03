from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers
import keras
from utils import *
from matplotlib import pyplot as plt
from IPython.display import clear_output
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import seaborn as sns

# sns.set(color_codes=True)

from keras import backend as K
def mean_euc_dist(y_true, y_pred):
    return K.mean(K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True)))

def func_model(f,nodes,lr):
    inputs = Input(shape=(X.shape[1],))

    # a layer instance is callable on a tensor, and returns a tensor
    x = Dense(nodes, activation=f)(inputs)
    predictions = Dense(1, activation=f)(x)

    # This creates a model that includes
    # the Input layer and two Dense layers
    model = Model(inputs=inputs, outputs=predictions)
    sgd = optimizers.SGD(lr=lr, decay=1e-2, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    return model

X, y = init(1, shuffle=True)
tanh = True
testX,testy=test_data(1)
lrs = [0.01,0.05,0.1]
hls = [5,10,15]
cvs = []
fig = plt.figure(figsize=(10, 10))
fig_dims = (3, 3)
sns.set()
assx=0

for hl in hls:
    assy=0
    for lr in lrs:


        if tanh:
            y = np.where(y == 0, -1, y)
            testy = np.where(testy == 0, -1, testy)
            model = func_model('tanh',5,lr)
        else: model = func_model('relu',5,lr)

        kfold = KFold(n_splits=3)
        cvscores = []

        for train, test in kfold.split(X, y):
        	# Fit the model
        	model.fit(X[train], y[train], epochs=100, verbose=0, shuffle=False)
        	# evaluate the model
        	scores = model.evaluate(X[test], y[test], verbose=0)
        	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        	cvscores.append(scores[1] * 100)
        print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
        cvs.append([np.mean(cvscores), np.std(cvscores), lr, hl])

        model = func_model('tanh',5,lr)

        history = model.fit(X,y, epochs=100, verbose=0, validation_data=(testX,testy), shuffle=False)


        plt.subplot2grid(fig_dims, (assx, assy))
        plt.xlabel('Epochs')
        plt.plot(history.history['acc'],label="acc_train")
        plt.plot(history.history['val_acc'], label='acc_test',linestyle='--')
        plt.legend(loc='best')



        assy=assy+1

    assx=assx+1

plt.show()
fig.savefig('kerasGrid.png')

m = max([b[0] for b in cvs])
best = next((x for x in cvs if m == x[0]), None)

print "-------------"
print "Best with lr=%.2f and %d hidden layer units:" % (best[2], best[3])
print "%.1f%% (+/- %.1f%%)" % (best[0], best[1])


#############################################################################



nodes = [2] # number of nodes in the hidden layer
lrs = [0.1] # learning rate, default = 0.001
epochs = 150
batch_size = 15

def build_model(nodes=10, lr=0.001):

    inputs = Input(shape=(X.shape[1],))
    x = Dense(nodes, activation='tanh')(inputs)
    predictions = Dense(1, activation='tanh')(x)

    # This creates a model that includes
    # the Input layer and two Dense layers
    model = Model(inputs=inputs, outputs=predictions)
    sgd = optimizers.SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=False)
    model.compile(optimizer=sgd,
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=build_model, epochs=epochs,
                        batch_size=batch_size, verbose=0)
param_grid = dict(nodes=nodes, lr=lrs)
param_grid

grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=10,
                    n_jobs=1, refit=True, verbose=0,return_train_score=True)
grid_result = grid.fit(X, y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']*100
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

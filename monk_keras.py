from keras.layers import Input, Dense
from keras.models import Model, Sequential
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
# import seaborn as sns
# sns.set(color_codes=True)

# updatable plot
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.accs = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):

        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.accs.append(logs.get('acc'))
        self.i += 1

        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.accs, label="acc")
        plt.legend()
        plt.show();
plot_losses = PlotLosses()

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
                  loss=mean_euc_dist,
                  metrics=['accuracy'])
    return model

X, y = init(4, shuffle=True)
tanh = False
lrs = [0.01,0.05,0.1]
hls = [3,5,15]
cvs = []
for hl in hls:
    for lr in lrs:
        if tanh:
            y = np.where(y == 0, -1, y)
            model = func_model('tanh',hl,lr)
        else: model = func_model('relu',hl,lr)
        kfold = KFold(n_splits=3)
        cvscores = []
        for train, test in kfold.split(X, y):
        	# Fit the model
        	model.fit(X[train], y[train], batch_size=256, epochs=250, verbose=0, shuffle=False)
        	# evaluate the model
        	scores = model.evaluate(X[test], y[test], verbose=0)
        	print("%s: %.2f" % (model.metrics_names[0], scores[0]))
        	cvscores.append(scores[0])
        print("%.2f (+/- %.2f)" % (np.mean(cvscores), np.std(cvscores)))
        cvs.append([np.mean(cvscores), np.std(cvscores), lr, hl])

m = min([b[0] for b in cvs])
best = next((x for x in cvs if m == x[0]), None)

print "-------------"
print "Best with lr=%.2f and %d hidden layer units:" % (best[2], best[3])
print "%.2f (+/- %.2f)" % (best[0], best[1])

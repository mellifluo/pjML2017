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

def baseline_model():
    # create model
    model = Sequential()
    model.add(Dense(5, input_dim=X.shape[1], activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))
    sgd = optimizers.SGD(lr=0.5, decay=0, momentum=0.9, nesterov=False)
    model.compile(loss='mean_squared_error', optimizer=sgd,metrics=['accuracy'])
    return model

def func_model():
    inputs = Input(shape=(X.shape[0],X.shape[1],))

    # a layer instance is callable on a tensor, and returns a tensor
    x = Dense(5, activation='sigmoid')(inputs)
    predictions = Dense(1, activation='sigmoid')(x)

    # This creates a model that includes
    # the Input layer and two Dense layers
    model = Model(inputs=inputs, outputs=predictions)
    sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=False)
    model.compile(optimizer=sgd,
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    X = X[np.newaxis, ...]
    y = y[np.newaxis, ...]
    return model

X, y = init()

# model.fit(X, y, epochs=300, callbacks=[plot_losses])  # starts training
# model.fit(X, y, epochs=100)  # starts training
# testX, testY = test_data()
# testX = testX[testX.shape[0] - 124:]
# testY = testY[testY.shape[0] - 124:]
# testX = testX[np.newaxis, :]
# testY = testY[np.newaxis, :]
# model.evaluate(x=testX, y=testY)

# fix random seed for reproducibility
seed = 17
np.random.seed(seed)
estimator = KerasClassifier(build_fn=baseline_model, nb_epoch=500, verbose=0)
kfold = KFold(n_splits=3, random_state=seed)
results = cross_val_score(estimator, X, y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

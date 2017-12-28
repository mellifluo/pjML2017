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

def func_model(f):
    inputs = Input(shape=(X.shape[1],))

    # a layer instance is callable on a tensor, and returns a tensor
    x = Dense(5, activation=f)(inputs)
    predictions = Dense(1, activation=f)(x)

    # This creates a model that includes
    # the Input layer and two Dense layers
    model = Model(inputs=inputs, outputs=predictions)
    sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=False)
    model.compile(optimizer=sgd,
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    return model

X, y = init(1, shuffle=True)
tanh = True
if tanh:
    y = np.where(y == 0, -1, y)
    model = func_model('tanh')
else: model = func_model('sigmoid')
kfold = KFold(n_splits=10)
cvscores = []
for train, test in kfold.split(X, y):
	# Fit the model
	model.fit(X[train], y[train], epochs=100, verbose=0)
	# evaluate the model
	scores = model.evaluate(X[test], y[test], verbose=0)
	print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
	cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))

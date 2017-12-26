from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers
import keras
from utils import *
from matplotlib import pyplot as plt
from IPython.display import clear_output
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

X, y = init()
# This returns a tensor
inputs = Input(shape=X.shape)

# a layer instance is callable on a tensor, and returns a tensor
x = Dense(5, activation='tanh')(inputs)
predictions = Dense(1, activation='tanh')(x)

# This creates a model that includes
# the Input layer and two Dense layers
model = Model(inputs=inputs, outputs=predictions)

sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd,
              loss='mean_squared_error',
              metrics=['accuracy'])
X = X[np.newaxis, ...]
y = y[np.newaxis, ...]
model.fit(X, y, epochs=100, callbacks=[plot_losses])  # starts training
testX, testY = test_data()
testX = testX[testX.shape[0] - 124:]
testY = testY[testY.shape[0] - 124:]
testX = testX[np.newaxis, :]
testY = testY[np.newaxis, :]
model.evaluate(x=testX, y=testY)

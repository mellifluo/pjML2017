from keras.layers import Input, Dense,Activation
from keras.models import Model
from keras import optimizers
import seaborn as sns
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
sns.set(color_codes=True)

def func_model(f,nodes,lr,mom):
    inputs = Input(shape=(X.shape[1],))

    # a layer instance is callable on a tensor, and returns a tensor
    x = Dense(nodes, activation=f)(inputs)
    predictions = Dense(1, activation=f)(x)

    # This creates a model that includes
    # the Input layer and two Dense layers
    model = Model(inputs=inputs, outputs=predictions)
    sgd = optimizers.SGD(lr=lr, decay=1e-3, momentum=mom, nesterov=True)
    model.compile(optimizer=sgd,
                  loss='mean_squared_error',
                  metrics=['accuracy'])
    return model

d = 1
X, y = init(d, shuffle=True)
kfold = KFold(n_splits=5)
cvscores = []
moms = [0.9,0.7,0.5]
lrs = [0.05,0.1,0.5]

fig = plt.figure(figsize=(50,50))
fig_dims = (10, 10)
assx=0
for hl in np.arange(5,16,5):
        assy=0
        for lr in lrs:

            model = func_model('tanh',hl,lr,0.9)

            for train, test in kfold.split(X, y):
            	# Fit the model
            	model.fit(X[train], y[train], epochs=250, verbose=0, shuffle=False)
            	# evaluate the model
            	scores = model.evaluate(X[test], y[test], verbose=0)
            	#print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            	cvscores.append(scores[1] * 100)


            print 'Accuracy con unita=%0.2f e lr=%0.2f: %0.2f (+/- %0.2f)' % (hl,lr,np.mean(cvscores), np.std(cvscores))

            assy=assy+1
            model = func_model('tanh',hl,lr,0.9)
            history= model.fit(X, y, epochs=250, verbose=0, shuffle=False)

            # evaluate the model

            plt.subplot2grid(fig_dims, (assx, assy))
            plt.plot(history.history['loss'])
            plt.plot(history.history['acc'])
        assx=assx+1
plt.show()

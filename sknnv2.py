from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from utils import *
import seaborn as sns
from sklearn.model_selection import *
from sklearn.metrics import make_scorer

def mean_euc_dist(y_true, y_pred):
    return np.mean(np.sqrt(np.sum(np.square(y_true - y_pred), axis=-1, keepdims=True)))

def nn(d=1,classification=True):
    if classification:
        d = 1
        X,y=init(d,True)
        y = np.where(y == 0, -1, y)
        c, r = y.shape
        y = y.reshape(c,)
        fig = plt.figure(figsize=(15, 15))
        fig_dims = (3, 3)
        sns.set()
        lrs = [0.05,0.1,0.5]
        moms = [0.9,0.7,0.5]
        assx=0
        accs = []
        for hl in np.arange(5,16,5):
                assy=0
                for lr in lrs:

                    clf = MLPClassifier(solver='sgd',tol=0,activation="tanh",learning_rate_init=lr,max_iter=100,nesterovs_momentum=True,
                                        hidden_layer_sizes=(hl,), loss='mean_squared',momentum=0.9, alpha=1e-3, batch_size=32, shuffle=False)
                    scores = cross_validate(clf,X,y,cv=3, scoring='accuracy')
                    plt.subplot2grid(fig_dims, (assx, assy))
                    plt.plot(clf.loss_curve_)
                    plt.plot(accs)
                    print clf.loss
                    assy=assy+1

                    print 'Accuracy con unita=%0.2f e lr=%0.2f: %0.2f (+/- %0.2f)' % (hl,lr,scores.mean(), scores.std() * 2)

                assx=assx+1

        plt.show()
    else:
        X,y=init(4,True)
        fig = plt.figure(figsize=(15, 15))

        fig_dims = (3, 3)
        sns.set()
        lrs = [0.1]
        moms = [0.9,0.7,0.5]
        assx=0
        for hl in np.arange(5,16,5):
                assy=0
                for lr in lrs:

                    clf = MLPRegressor(solver='sgd',tol=0,activation="identity",learning_rate_init=lr,max_iter=250,nesterovs_momentum=True,
                                        hidden_layer_sizes=(hl,), verbose=True, momentum=0.9, loss='euclid_dist', alpha=1e-2, batch_size=32, shuffle=False)
                    scorer = make_scorer(mean_euc_dist, greater_is_better=False)
                    scores = cross_val_score(clf,X,y,cv=5)
                    clf.fit(X, y)
                    plt.subplot2grid(fig_dims, (assx, assy))
                    plt.plot(clf.loss_curve_)
                    assy=assy+1

                    print 'Accuracy con unita=%0.2f e lr=%0.2f: %0.2f (+/- %0.2f)' % (hl,lr,-1*scores.mean(), scores.std() * 2)
                assx=assx+1

        plt.show()



nn(classification=True)

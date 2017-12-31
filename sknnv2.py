from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from utils import *
import seaborn as sns
from sklearn.model_selection import cross_val_score



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
        for hl in np.arange(5,16,5):
                assy=0
                for lr in lrs:

                    clf = MLPClassifier(solver='sgd',tol=0,activation="tanh",learning_rate_init=lr,max_iter=500,nesterovs_momentum=True,verbose=1,
                                        hidden_layer_sizes=(hl,), loss='mean_squared',momentum=0.9, alpha=1e-3, batch_size=32, shuffle=False)

                    scores = cross_val_score(clf,X,y,cv=10)
                    clf.fit(X, y )
                    plt.subplot2grid(fig_dims, (assx, assy))
                    plt.plot(clf.loss_curve_)
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
        lrs = [0.05,0.1,0.5]
        moms = [0.9,0.7,0.5]
        assx=0
        clf = MLPRegressor(solver='sgd',tol=0,activation="identity",learning_rate_init=0.01,max_iter=100,nesterovs_momentum=False,
                            hidden_layer_sizes=(5,),loss='euclid_dist', batch_size=32, momentum=0.9, alpha=0.01, shuffle=False, verbose=True,learning_rate="adaptive")
        scores = cross_val_score(clf,X,y,cv=5)

        clf.fit(X, y)
        clf.loss
        plt.plot(clf.loss_curve_)
        plt.show()
        for hl in np.arange(5,16,5):
                assy=0
                for lr in lrs:

                    clf = MLPRegressor(solver='sgd',tol=0,activation="relu",learning_rate_init=lr,max_iter=1000,nesterovs_momentum=True,
                                        hidden_layer_sizes=(100,), momentum=0.9, alpha=1e-3, batch_size=32, shuffle=False)
                    scores = cross_val_score(clf,X,y,cv=5)
                    clf.fit(X, y)
                    plt.subplot2grid(fig_dims, (assx, assy))
                    plt.plot(clf.loss_curve_)
                    assy=assy+1

                    print 'Accuracy con unita=%0.2f e lr=%0.2f: %0.2f (+/- %0.2f)' % (hl,lr,scores.mean(), scores.std() * 2)

                assx=assx+1

        plt.show()



nn(classification=False)

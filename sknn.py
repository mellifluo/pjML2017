from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from utils import *
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import learning_curve

def mean_euc_dist(y_true, y_pred):
    return np.mean(np.sqrt(np.sum(np.square(y_true - y_pred), axis=-1, keepdims=True)))

def predict_regression(testX):
    clf = MLPRegressor(solver='sgd',tol=0,activation="tanh",learning_rate_init=0.01,max_iter=500,nesterovs_momentum=True,
                        hidden_layer_sizes=(3,), verbose=False, momentum=0.9, loss='euclid_dist', alpha=1e-3, batch_size=32, shuffle=False)
    clf.fit(X,y)
    res=clf.predict(testX)
    np.savetxt("scikit_cupresults.csv", res, delimiter=",")

def sk_nn(X,y,d=1,classification=True):
    if classification:
        y = np.where(y == 0, -1, y)
        c, r = y.shape
        y = y.reshape(c,)
        fig = plt.figure(figsize=(10,10))
        fig_dims = (3, 3)
        sns.set()
        lrs = [0.01,0.1,0.5]
        hls = [3,5,10]
        moms = [0.9,0.7,0.5]
        assx=0
        for hl in hls:
                assy=0
                for lr in lrs:

                    clf = MLPClassifier(solver='sgd',tol=0,activation="tanh",learning_rate_init=lr,max_iter=100,nesterovs_momentum=True,verbose=0,
                                        hidden_layer_sizes=(hl,), loss='mean_squared',momentum=0.9, alpha=0, batch_size=32, shuffle=False)

                    scores = cross_val_score(clf,X,y,cv=5)
                    clf.fit(X, y )
                    plt.subplot2grid(fig_dims, (assx, assy))
                    plt.xlabel('Epochs')
                    plt.plot(clf.loss_curve_,label="errore",linestyle='--')
                    plt.legend(loc='best')
                    assy=assy+1

                    print 'Accuracy con unita=%0.2f e lr=%0.2f: %0.2f (+/- %0.2f)' % (hl,lr,scores.mean(), scores.std() * 2)

                assx=assx+1

        plt.show()
    else:
        fig = plt.figure(figsize=(15, 15))
        fig_dims = (3, 3)
        sns.set()
        moms = [0.9,0.7,0.5]
        alphas = [0.01,0.001,0.0001]
        lrs = [0.01,0.05,0.1]
        hls = [3,5,10]
        testX,testy=test_data()
        assx=0
        for hl in hls:
                assy=0
                for lr in lrs:

                    clf = MLPRegressor(solver='sgd',tol=0,activation="tanh",learning_rate_init=lr,max_iter=100,nesterovs_momentum=True,
                                        hidden_layer_sizes=(hl,), verbose=False, momentum=0.9, loss='euclid_dist', alpha=1e-3, batch_size=32, shuffle=False)
                    scorer = make_scorer(mean_euc_dist, greater_is_better=False)
                    scores = cross_val_score(clf,X,y,cv=3, scoring=scorer)
                    clf.fit(X,y)
                    plt.subplot2grid(fig_dims, (assx, assy))
                    plt.xlabel('Epochs')
                    plt.plot(clf.loss_curve_,label="errore",linestyle='--')
                    plt.legend(loc='best')
                    plt.plot(clf.loss_curve_)

                    assy=assy+1

                    print 'Accuracy con unita=%0.2f e alphas=%0.4f: %0.2f (+/- %0.2f)' % (hl,lr,-1*scores.mean(), scores.std() * 2)
                assx=assx+1



        plt.show()

        predict_regression(testX)

d = 1
X,y=init(d,True)
sk_nn(X,y,d,classification=True)
# a = np.loadtxt('scikit_cupresults.csv', dtype='float', delimiter=',')

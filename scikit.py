from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from utils import *
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer

# define MEE
def mean_euc_dist(y_true, y_pred):
    return np.mean(np.sqrt(np.sum(np.square(y_true - y_pred), axis=-1, keepdims=True)))

# predict regression for the cup
def predict_regression(testX):
    clf = MLPRegressor(solver='sgd',tol=0,activation="tanh",learning_rate_init=0.01,max_iter=500,nesterovs_momentum=True,
                        hidden_layer_sizes=(10,), verbose=False, momentum=0.9, loss='euclid_dist', alpha=1e-6, batch_size=32, shuffle=False)
    clf.fit(X,y)
    res=clf.predict(testX)
    np.savetxt("scikit_cupresults.csv", res, delimiter=",")

# define the model
def sk_nn(X,y,testX,testy,d=1,classification=True):
    if classification:
        y = np.where(y == 0, -1, y)
        testy = np.where(testy == 0, -1, testy)
        # needed to use mlpclassifier
        c, r = y.shape
        y = y.reshape(c,)
        fig = plt.figure(figsize=(10,10))
        fig_dims = (3, 3)
        sns.set()
        lrs = [0.01,0.05,0.09]
        moms = [0.9,0.7,0.5]
        hls=[3,7,10]
        assx=0
        for hl in hls:
                assy=0
                for lr in lrs:
                    # define model
                    # we insert loss as a parameter modifying source code
                    clf = MLPClassifier(solver='sgd',tol=0,activation="tanh",learning_rate_init=lr,max_iter=100,nesterovs_momentum=True,verbose=0,
                                        hidden_layer_sizes=(hl,), loss='mean_squared',momentum=0.9, alpha=1e-6, batch_size=32, shuffle=False)
                    # cross validation
                    scores = cross_val_score(clf,X,y,cv=10)
                    clf.fit(X, y)
                    # mse of the test
                    print clf.score(testX,testy)
                    print clf.loss_
                    # plot for the grid search
                    plt.subplot2grid(fig_dims, (assx, assy))
                    plt.xlabel('Epochs')
                    plt.plot(clf.loss_curve_,label="errore",linestyle='--')
                    plt.legend(loc='best')
                    assy=assy+1
                    # result of cross val
                    print 'Accuracy con unita=%0.2f e lr=%0.2f: %0.2f (+/- %0.2f)' % (hl,lr,scores.mean(), scores.std() * 2)
                assx=assx+1
        plt.show()
    else:
        fig = plt.figure(figsize=(8, 8))
        fig_dims = (3, 3)
        sns.set()
        moms = [0.9,0.7,0.5]
        alphas = [0.01,0.001,0.0001]
        lrs = [0.01,0.05,0.1]
        hls = [4,7,10]
        testX,testy=test_data()
        assx=0
        for hl in hls:
                assy=0
                for lr in lrs:
                    # regressor model
                    clf = MLPRegressor(solver='sgd',tol=0,activation="tanh",learning_rate_init=lr,max_iter=100,nesterovs_momentum=True,
                                        hidden_layer_sizes=(hl,), verbose=False, momentum=0.9, loss='euclid_dist', alpha=1e-3, batch_size=32, shuffle=False)
                    # define a scorer to  the error func we need to evaluate
                    scorer = make_scorer(mean_euc_dist, greater_is_better=False)
                    scores = cross_val_score(clf,X,y,cv=3, scoring=scorer)
                    # time in computation
                    time_start = time.clock()
                    clf.fit(X,y)
                    print (time.clock() - time_start)
                    # plotting grid search
                    plt.subplot2grid(fig_dims, (assx, assy))
                    plt.tight_layout()
                    plt.title('lr='+ str(lr)+" U="+str(hl))
                    plt.xlabel('Epochs')
                    plt.plot(clf.loss_curve_,label="MEE",linestyle='-')
                    plt.legend(loc='best')
                    plt.plot(clf.loss_curve_)
                    print assx,assy
                    assy=assy+1
                    # results of crossval
                    print 'Accuracy con unita=%0.2f e lr=%0.3f: %0.2f (+/- %0.2f)' % (hl,lr,-1*scores.mean(), scores.std() * 2)
                assx=assx+1
        plt.show()
        fig.savefig('./gridReg.png')
        predict_regression(testX)

# settings
d=4
X,y=init(d,True)
testX,testy=test_data(d)

# command to run
# sk_nn(X,y,testX,testy,d,classification=False)

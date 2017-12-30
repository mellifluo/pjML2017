from sklearn.neural_network import MLPClassifier
from utils import *
import seaborn as sns
from sklearn.model_selection import cross_val_score

d=2
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

            clf = MLPClassifier(solver='sgd',tol=0,activation="tanh",learning_rate_init=lr,max_iter=250,nesterovs_momentum=True,
                                hidden_layer_sizes=(hl,), momentum=0.9, alpha=1e-3, batch_size=32, shuffle=False)
            scores = cross_val_score(clf,X,y,cv=5)
            clf.fit(X, y)
            plt.subplot2grid(fig_dims, (assx, assy))
            plt.plot(clf.loss_curve_)
            assy=assy+1

            print 'Accuracy con unita=%0.2f e lr=%0.2f: %0.2f (+/- %0.2f)' % (hl,lr,scores.mean(), scores.std() * 2)

        assx=assx+1

plt.show()

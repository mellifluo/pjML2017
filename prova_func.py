from nn import *

X, y = init()

for eta in np.arange(0.1,1,0.1):
    nn(X, y, lr=eta, fit=True)
    

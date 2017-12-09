from nn import *

X, y = init()

for hl in range(3,7):
    for eta in np.arange(0.1,1,0.1):
        nn(X, y, lr=eta, hl_u=hl, fit=True, epoch=200)

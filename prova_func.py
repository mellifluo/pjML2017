from nn import *

"""
file per fare tutte 'e prov e ripigliare tutt' chell che è nuost
"""

X, y = init()

for hl in range(3,4):
    for eta in np.arange(0.1,1,0.3):
        nn(X, y, lr=eta, hl_u=hl, beta=0.001, fit=True, epoch=200, tanh=True)

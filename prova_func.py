from nn import *

"""
file per fare tutte 'e prov e ripigliare tutt' chell che è nuost
"""

X, y = init()

# questo se vuoi fare piu prove contemporaneamente (occhio che puo venire un macello)
for hl in range(12,15):
    for eta in np.arange(0.1,1,0.3):
        nn(X, lr=eta, hl_u=hl, fit=True, epoch=200, tanh=True)

# basta questo (aggiungendo i parametri che vuoi)
nn(X, epoch=500)

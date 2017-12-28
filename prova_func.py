from nn import *

"""
file per fare tutte 'e prov e ripigliare tutt' chell che e nuost
"""
d=1
X, y = init()

# questo se vuoi fare piu prove contemporaneamente (occhio che puo venire un macello)
for hl in range(1,11):
    if hl == 3 or hl == 5 or hl == 10:
        for eta in np.arange(0.1,1,0.3):
            nn(X,y, d, lr=eta, hl_u=hl, fit=True, epoch=500, tanh=True, cv=3)

# basta questo (aggiungendo i parametri che vuoi)
# nn(X, epoch=500)

"""
9. Target Concepts associated to the MONK's problem:

   MONK-1: (a1 = a2) or (a5 = 1)

   MONK-2: EXACTLY TWO of {a1 = 1, a2 = 1, a3 = 1, a4 = 1, a5 = 1, a6 = 1}

   MONK-3: (a5 = 3 and a4 = 1) or (a5 /= 4 and a2 /= 3)
           (5percent class noise added to the training set)
"""

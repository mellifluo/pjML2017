from utils import *
from nn import *

"""
file per fare tutte 'e prov e ripigliare tutt' chell che e nuost
"""
d=1
X, y = init(d, shuffle=True)
hls = [5,10,15]
lrs = [0.05,0.1,0.5]
alphas = [0.01,0.005,0.001]
moms = [0.9,0.7,0.5]

# questo se vuoi fare piu prove contemporaneamente (occhio che puo venire un macello)
cvs = []
for hl in hls:
    for lr in lrs:
        cvs.append(nn(X, y, d, lr=lr, hl_u=hl, bs=32, epoch=250, mom=0.9, tanh=True, alpha=1e-3, cv=5))
# forse basta il massimo dell acc_val
m = max([b[0] for b in cvs])
best = next((x for x in cvs if m == x[0]), None)

print "-------------"
print "Best with lr=%.2f and %d hidden layer units:" % (best[2], best[3])
print "%.1f%% (+/- %.1f%%)" % (best[0], best[1])

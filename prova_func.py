from utils import *
from nn import *

"""
file per fare tutte 'e prov e ripigliare tutt' chell che e nuost
"""
d=4
hls = [3,5,15]
lrs = [0.05,0.1,0.5]
alphas = [0.01,0.001,0.0001]
moms = [0.9,0.7,0.5]

def gridsearch(setA, lr_fix=0.1, alpha_fix=1e-3, mom_fix=0.9):
    """
    gridsearch with hidden layer units set
    setA: could be lrs, alphas or moms(momentums)
    """
    cvs = []
    if setA is lrs:
        s = "lr"
        for hl in hls:
            for a in setA:
                cvs.append(nn(d, bs=256, epoch=250, lr=a, hl_u=hl, mom=mom_fix, cv=3, alpha=alpha_fix))
    elif setA is alphas:
        s = "a"
        for hl in hls:
            for a in setA:
                cvs.append(nn(d, bs=256, epoch=250, lr=lr_fix, hl_u=hl, mom=mom_fix, cv=3, alpha=a))
    elif setA is moms:
        s = "mom"
        for hl in hls:
            for a in setA:
                cvs.append(nn(d, bs=256, epoch=250, lr=lr_fix, hl_u=hl, mom=a, cv=3, alpha=alpha_fix))
    # forse basta il massimo dell acc_val
    if d == 4:
        m = min([b[0] for b in cvs])
        best = next((x for x in cvs if m == x[0]), None)
        print "-------------"
        print "Best with "+s+"=%.2f and %d hidden layer units:" % (best[2], best[3])
        print "%.1f (+/- %.1f)" % (best[0], best[1])
    else:
        m = max([b[0] for b in cvs])
        best = next((x for x in cvs if m == x[0]), None)
        print "-------------"
        print "Best with "+s+"=%.2f and %d hidden layer units:" % (best[2], best[3])
        print "%.1f%% (+/- %.1f%%)" % (best[0], best[1])

gridsearch(lrs, lr_fix=0.01)

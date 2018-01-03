from utils import *
from nn import *

"""
file per fare tutte 'e prov e ripigliare tutt' chell che e nuost
"""
hls = [5,10,15]
lrs = [0.05,0.1,0.5]
alphas = [0.01,0.001,0.0001]
moms = [0.9,0.7,0.5]

def gridsearch(setA, d=4, lr_fix=0.1, alpha_fix=1e-3, mom_fix=0.9, fit=False):
    """
    gridsearch with hidden layer units set
    setA: could be lrs, alphas or moms(momentums)
    """
    cvs = []
    if setA is lrs:
        s = "lr"
        for hl in hls:
            for a in setA:
                cvs.append(nn(d, bs=32, epoch=100, lr=a, hl_u=hl, mom=mom_fix, cv=3, alpha=alpha_fix))
    elif setA is alphas:
        s = "a"
        for hl in hls:
            for a in setA:
                cvs.append(nn(d, bs=32, epoch=100, lr=lr_fix, hl_u=hl, mom=mom_fix, cv=3, alpha=a))
    elif setA is moms:
        s = "mom"
        for hl in hls:
            for a in setA:
                cvs.append(nn(d, bs=32, epoch=100, lr=lr_fix, hl_u=hl, mom=a, cv=3, alpha=alpha_fix))
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
    if fit:
        if setA is lrs:
            nn(d, bs=32, epoch=100, lr=best[2], hl_u=best[3], mom=mom_fix, cv=None, alpha=alpha_fix)
        elif setA is alphas:
            nn(d, bs=32, epoch=100, lr=lr_fix, hl_u=best[3], mom=mom_fix, cv=None, alpha=best[2])
        elif setA is moms:
            nn(d, bs=32, epoch=100, lr=lr_fix, hl_u=best[3], mom=best[2], cv=None, alpha=alpha_fix)


gridsearch(lrs, d=2, alpha_fix=1e-2, fit=True)

from utils import *
from tf_model import *

"""
change parameters for gridsearch
"""
hls = [3,4,5]
lrs = [0.01,0.05,0.1]
alphas = [0.01,0.001,0.0001]
moms = [0.9,0.7,0.5]

def gridsearch(setA, d=4, lr_fix=0.1, alpha_fix=1e-3, mom_fix=0.9, fit=False):
    cvs = []
    # search in terms of global arrays (lrs,alphas,moms). hls always fixed
    if setA is lrs:
        s = "lr"
        for hl in hls:
            for a in setA:
                cvs.append(nn(d, bs=32, epoch=200, lr=a, hl_u=hl, mom=mom_fix, cv=5, alpha=alpha_fix))
    elif setA is alphas:
        s = "a"
        for hl in hls:
            for a in setA:
                cvs.append(nn(d, bs=32, epoch=200, lr=lr_fix, hl_u=hl, mom=mom_fix, cv=5, alpha=a))
    elif setA is moms:
        s = "mom"
        for hl in hls:
            for a in setA:
                cvs.append(nn(d, bs=32, epoch=200, lr=lr_fix, hl_u=hl, mom=a, cv=5, alpha=alpha_fix))
    # (sort of) model selection (a really greedy choice, not used for the project)
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
            nn(d, bs=32, epoch=500, lr=best[2], hl_u=best[3], mom=mom_fix, cv=None, alpha=alpha_fix)
        elif setA is alphas:
            nn(d, bs=32, epoch=500, lr=lr_fix, hl_u=best[3], mom=mom_fix, cv=None, alpha=best[2])
        elif setA is moms:
            nn(d, bs=32, epoch=500, lr=lr_fix, hl_u=best[3], mom=best[2], cv=None, alpha=alpha_fix)

# for model selection:
# gridsearch(lrs, d=4, alpha_fix=1e-6, fit=True)
###################################
# if model selection was already made, run only this command, with cv=None:
# nn(4, bs=32, epoch=100, lr=0.01, alpha=1e-3, hl_u=10, tanh=False, cv=None)

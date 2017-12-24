from utils import *

# funziona solo con k=3 (3-fold)
def cross_validation(x, k):
    test = np.loadtxt('./monk/monks-1.test', dtype='string', delimiter=' ')
    test = clean_data(test)
    test, y = split_classes(test)
    test = norm_data(test)
    test = np.append(y, test, axis=1)
    l = len(test)
    # in pratica faccio un tensore in cui concateno le parti di tr,val e ts.
    # di fatto non so se e la cosa giusta, anche perche se si vuole fare per bene
    # si deve prendere le tre partizioni con dimensioni diverse e quindi fanculo
    d = np.zeros(shape=(k,l/k,test.shape[1]))
    for i in range(0,k):
        d[i,:,:] = test[:l/k,:]
        test = test[l/k:,:]
    return d

# scambia 3 set
def three_switch(x,y,z):
    tmp = x
    x = y
    y = z
    z = tmp
    return x, y, z

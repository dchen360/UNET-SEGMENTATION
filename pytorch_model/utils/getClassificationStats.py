# Get Classification statistics (error, fpr, fnr)
# Felipe Giuste
# 07/30/2020

import numpy as np


## Get Classification Stats ##
def getStats(y_hat, y):
    #     pred = np.array( y_hat.cpu().data[:,0] ) > 0.5
    pred = np.array(y_hat) > 0.5
    real = np.array(y)
    neq = np.not_equal(pred, real)
    err = float(neq.sum()) / pred.shape[0]
    fpr = float(np.logical_and(pred == 1, neq).sum()) / (real == 0).sum()
    fnr = float(np.logical_and(pred == 0, neq).sum()) / (real == 1).sum()
    return neq, err, fpr, fnr

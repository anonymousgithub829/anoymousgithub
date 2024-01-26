import numpy as np
from collections import defaultdict
import numba as nb
from time import time

@nb.njit('int32[:,::1](float64[:,::1])', parallel=True)
def fastSort(a):
    b = np.empty(a.shape, dtype=np.int32)
    for i in nb.prange(a.shape[0]):
        b[i,:] = np.argsort(a[i,:])
    return b

def evaluate11(posprobe, r, k):
    userlist = list(posprobe.keys())

    r[userlist, :] = fastSort(r[userlist, :])
    pred = r[:, ::-1][:, 0:k[-1]]

    recall = []
    precision = []
    map = []
    ndcg = []
    for kk in k:
        recall_tmp = []
        precision_tmp = []
        map_tmp = []
        ndcg_tmp = []
        rank = np.arange(1, kk + 1)
        scores = np.log2(np.arange(2, kk + 2))
        for user in userlist:
            predict_tmp = np.zeros(kk, dtype=np.float)
            boo_tmp = np.zeros(kk, dtype=np.float)
            num_pos = len(posprobe[user])
            max_r = np.minimum(kk, num_pos)
            max_r_vector = np.zeros(kk, dtype=np.float)
            max_r_vector[:max_r] = 1
            ll = 1
            for l in range(kk):
                if pred[user, l] in posprobe[user]:
                    predict_tmp[l] = ll
                    boo_tmp[l] = 1
                    ll += 1
            sum_tmp = np.sum(boo_tmp)
            recall_tmp.append(sum_tmp / num_pos)
            precision_tmp.append(sum_tmp / kk)
            map_tmp.append(np.sum(predict_tmp / rank) / kk)
            idcg = np.sum(max_r_vector/scores)
            dcg = np.sum(boo_tmp/scores)
            ndcg_tmp.append(dcg/idcg)

        recall.append(np.mean(recall_tmp))
        precision.append(np.mean(precision_tmp))
        map.append(np.mean(map_tmp))
        ndcg.append(np.mean(ndcg_tmp))

    return recall, precision, map, ndcg




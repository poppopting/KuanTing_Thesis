def predict_svd(model, data):
    pre_list = []
    for i in range(0, len(data), 500):
        user_s, item_s, ratings = data[i: i+500].values.T.astype('uint16')
        pre_list.extend(model.get_rating(user_s, item_s).tolist())
        
    return pre_list
def predict_ord(model, data):
    pre_list = []
    for i in range(0, len(data), 500):
        user_s, item_s, ratings = data[i: i+500].values.T.astype('uint16')
        tmp = model.get_rating(user_s, item_s)[1].argmax(axis=1) + 1
        pre_list.extend(tmp.tolist())
        
    return pre_list

from collections import defaultdict
import numpy as np
def FCP1(real_data, predictions):


    predictions_u = defaultdict(list)
    nc_u = defaultdict(int)
    nd_u = defaultdict(int)

    for (u0, _, r0), est in zip(real_data.values, predictions):
        predictions_u[u0].append((r0, est))

    for u0, preds in predictions_u.items():
        for r0i, esti in preds:
            for r0j, estj in preds:
                con1 = (esti > estj) and (r0i > r0j)
                con2 = (esti == estj) and (r0i == r0j)
                con3 = (esti < estj) and (r0i < r0j)
                if con1 | con2 | con3:
                    nc_u[u0] += 1
                
                con1 = (esti >= estj) and (r0i < r0j)
                con2 = (esti <= estj) and (r0i > r0j)
                if con1 | con2 :
                    nd_u[u0] += 1
    
    nc = np.sum(list(nc_u.values())) if nc_u else 0
    nd = np.sum(list(nd_u.values())) if nd_u else 0

    try:
        fcp = nc / (nc + nd)
    except ZeroDivisionError:
        raise ValueError('cannot compute fcp on this list of prediction. ' +
                         'Does every user have at least two predictions?')

    return fcp
def FCP2(real_data, predictions):

    predictions_u = defaultdict(list)
    nc_u = defaultdict(int)
    nd_u = defaultdict(int)

    for (u0, _, r0), est in zip(real_data.values, predictions):
        predictions_u[u0].append((r0, est))

    for u0, preds in predictions_u.items():
        for r0i, esti in preds:
            for r0j, estj in preds:
                if esti > estj and r0i > r0j:
                    nc_u[u0] += 1
                if esti >= estj and r0i < r0j:
                    nd_u[u0] += 1

    nc = np.sum(list(nc_u.values())) if nc_u else 0
    nd = np.sum(list(nd_u.values())) if nd_u else 0

    try:
        fcp = nc / (nc + nd)
        
    except ZeroDivisionError:
        raise ValueError('cannot compute fcp on this list of prediction. ' +
                         'Does every user have at least two predictions?')


    return fcp
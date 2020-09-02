import numpy as np
import numpy_indexed as npi
def check_svd_gradient(model, train, reg):

    users, items, trues = train.values.T
    full_mat = model.full_matrix()
    predict = np.asarray([full_mat[i-1, j-1] for i, j in zip(users, items)])
    data_len = len(train)
    e = trues - predict
    
    p_tmp = -e[:, np.newaxis] * model.Q[items-1, :]
    uniq_user_p, p_gradient = npi.group_by(users).sum(p_tmp, axis=0)
    p_gradient = p_gradient / data_len + reg * model.P
    
    q_tmp = -e[:, np.newaxis] * model.P[users-1, :]
    uniq_item_q, q_gradient = npi.group_by(items).sum(q_tmp, axis=0)
    q_gradient = q_gradient / data_len + reg * model.Q[uniq_item_q-1, :]
    
    uniq_user_bu, bu_gradient = npi.group_by(users).sum(-e)
    bu_gradient = bu_gradient / data_len + reg * model.b_u
    
    uniq_item_bi, bi_gradient = npi.group_by(items).sum(-e)
    bi_gradient = bi_gradient / data_len + reg * model.b_i[uniq_item_bi-1]
    
    return (uniq_user_p, p_gradient), (uniq_item_q, q_gradient), (uniq_user_bu, bu_gradient), (uniq_item_bi, bi_gradient)
import numpy as np
import numpy_indexed as npi
def check_svdpp_gradient(model, train, reg):

    users, items, trues = train.values.T
    full_mat = model.full_matrix()
    predict = np.asarray([full_mat[i-1, j-1] for i, j in zip(users, items)])
    data_len = len(train)
    e = trues - predict
    
    p_tmp = -e[np.newaxis, :] * model.Q[:, items-1]
    uniq_user_p, p_gradient = npi.group_by(users).sum(p_tmp, axis=1)
    p_gradient = p_gradient / data_len + reg * model.P
    
    tmp = {user:model.P[:, user-1] + np.sqrt(1/len(model.users_rated[user]))*model.x[:, model.users_rated[user]-1].sum(axis=1)
                    for user in np.unique(users)}
    batch_tmp = np.concatenate([tmp[user][:, np.newaxis] for user in users], axis=1)
    q_tmp = -e[np.newaxis, :] * batch_tmp
    uniq_item_q, q_gradient = npi.group_by(items).sum(q_tmp, axis=1)
    q_gradient = q_gradient / data_len + reg * model.Q[:, uniq_item_q-1]
    
    uniq_user_bu, bu_gradient = npi.group_by(users).sum(-e)
    bu_gradient = bu_gradient / data_len + reg * model.b_u
    
    uniq_item_bi, bi_gradient = npi.group_by(items).sum(-e)
    bi_gradient = bi_gradient / data_len + reg * model.b_i[uniq_item_bi-1]
    
    update_x = np.zeros(model.Q.shape) 
    xjss = []
    for error, i, j in zip(e, users, items):
        rated = model.users_rated[i] 
        update_x[:, rated-1] += (error * np.sqrt(1/len(rated)) * model.Q[:, j-1])[:, np.newaxis]   

        xjss.append(rated)   

    xj_ravel = np.hstack(xjss)           
    unique_xj = np.unique(xj_ravel)
 #   update_x[:, :max(xj_ravel)] /= np.bincount(xj_ravel-1).clip(1)[np.newaxis, :]
    update_x = update_x / data_len + reg * update_x
   
    
    return (uniq_user_p, p_gradient), (uniq_item_q, q_gradient), (uniq_user_bu, bu_gradient), (uniq_item_bi, bi_gradient), (unique_xj, update_x)
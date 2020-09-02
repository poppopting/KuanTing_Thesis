import numpy as np
import numpy_indexed as npi
def check_ordmm_gradient(model, train, reg_dict):

    users, items, trues = train.values.T
    full_mat = model.full_matrix()[1]
    prediction = model.full_matrix()[2]
    data_len = len(train)
    cdfs_s = np.asarray([full_mat[i-1, j-1, :] for i, j in zip(users, items)])

    bigger_r = 1 - cdfs_s[np.arange(data_len), trues]
    smaller_r = cdfs_s[np.arange(data_len), trues-1]
    diff = bigger_r - smaller_r
    # bu
    unique_i, bu_grad = npi.group_by(users).sum(-diff, axis=0)
    bu_grad = bu_grad / data_len - reg_dict['reg_bu'] * model.b_u
    # bi
    unique_j, bi_grad = npi.group_by(items).sum(-diff, axis=0)
    bi_grad = bi_grad / data_len - reg_dict['reg_bi'] * model.b_i[unique_j-1]
    # P
    tmp_p = -model.Q[:, items-1] * diff
    unique_i, p_grad = npi.group_by(users).sum(tmp_p, axis=1)
    p_grad = p_grad / data_len - reg_dict['reg_pu'] * model.P
    # Q
    tmp_q = -model.P[:, users-1] * diff
    unique_j, q_grad = npi.group_by(items).sum(tmp_q, axis=1)
    q_grad = q_grad / data_len - reg_dict['reg_qi'] * model.Q[:, unique_j-1]
    
    t_grad = np.sum(diff)/ data_len - reg_dict['reg_t'] * model.t 
    t = model.t
    
    beta1, beta2, beta3 = model.beta.copy()
    # update_beta1
    update_beta1 =  np.sum(-cdfs_s[trues==5, 4]) + np.sum(diff[trues==3]) + np.sum(diff[trues==4]) + \
                    np.sum(cdfs_s[trues==2, 2]*(1-cdfs_s[trues==2, 2])/(cdfs_s[trues==2, 2]-cdfs_s[trues==2, 1]))
    update_beta1 *= np.exp(beta1)
    # update_beta2
    update_beta2 =  np.sum(cdfs_s[trues==3, 3]*(1-cdfs_s[trues==3, 3])/(cdfs_s[trues==3, 3]-cdfs_s[trues==3, 2])) + \
                    np.sum(diff[trues==4]) + np.sum(-cdfs_s[trues==5, 4])
    update_beta2 *= np.exp(beta2)
    # update_beta3
    update_beta3 =  np.sum(cdfs_s[trues==4, 4]*(1-cdfs_s[trues==4, 4])/(cdfs_s[trues==4, 4]-cdfs_s[trues==4, 3])) + \
                    np.sum(-cdfs_s[trues==5, 4])
    update_beta3 *= np.exp(beta3)

    beta_gradient = np.asarray([update_beta1, update_beta2, update_beta3])
    beta_gradient /= data_len
    beta_gradient -= reg_dict['reg_beta'] * model.beta

    return bu_grad, bi_grad, p_grad, q_grad, t_grad, (beta_gradient[0], beta_gradient[1], beta_gradient[2])
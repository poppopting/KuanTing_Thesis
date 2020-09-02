import os
import sys
import pickle
package_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(package_path)
from Ting_MF import ting_svd, ting_svdpp, ting_ord, ting_ordmm
from Ting_MF import check_svd_gradient, check_svdpp_gradient, check_ord_gradient, check_ordmm_gradient

import pandas as pd
import numpy as np

def convert_input(model, df):
    data = df.copy()
    data.loc[:, 'user'] = data.user.map(model.user2idx)
    data.loc[:, 'item'] = data.item.map(model.item2idx)
    
    return data

train_100k = pd.read_csv(r'ml-100k\train_100k.csv')

# take dim=100 to compute gradient values
dim = 100


################# SVD ###########################
weight_decay= 0.0004
model_svd = pickle.load(open('models\model_svd_ml100k_f%d.pkl'%dim, 'rb'))
(_, p_grad), (_, q_grad), (_, bu_grad), (_, bi_grad) = check_svd_gradient.check_svd_gradient(model_svd, 
                                                                                             convert_input(model_svd,
                                                                                                           train_100k),
                                                                                             weight_decay)
abs_p, abs_q, abs_bu, abs_bi = abs(p_grad), abs(q_grad), abs(bu_grad), abs(bi_grad)
svd_grad = [np.max(abs_p), np.min(abs_p), np.mean(p_grad),
            np.max(abs_q), np.min(abs_q), np.mean(q_grad),
            np.max(abs_bu), np.min(abs_bu), np.mean(bu_grad),
            np.max(abs_bi), np.min(abs_bi), np.mean(bi_grad),
            '', '', '', '', '', '', '']

################# SVDpp ###########################
weight_decay= 0.0004
model_svdpp = pickle.load(open('models\model_svdpp_ml100k_f%d.pkl'%dim, 'rb'))
(_, p_grad), (_, q_grad), (_, bu_grad), (_, bi_grad), (_, x_grad) = check_svdpp_gradient.check_svdpp_gradient(
                                                                                                     model_svdpp, 
                                                                                                     convert_input(model_svdpp,
                                                                                                                   train_100k),
                                                                                                     weight_decay)
abs_p, abs_q, abs_bu, abs_bi, abs_x = abs(p_grad), abs(q_grad), abs(bu_grad), abs(bi_grad), abs(x_grad) 
svdpp_grad = [np.max(abs_p), np.min(abs_p), np.mean(p_grad),
              np.max(abs_q), np.min(abs_q), np.mean(q_grad),
              np.max(abs_bu), np.min(abs_bu), np.mean(bu_grad),
              np.max(abs_bi), np.min(abs_bi), np.mean(bi_grad),
              np.max(abs_x), np.min(abs_x), np.mean(x_grad),
              '', '', '', '']

################# Ordmm ###########################
reg_dict = {'reg_bu': 0.001,
            'reg_bi': 0.001,
            'reg_pu': 0.001,
            'reg_qi': 0.001,
            'reg_t': 0.001,
            'reg_beta': 0.001}
model_ordmm = pickle.load(open('models\model_ordmm_ml100k_f%d.pkl'%dim, 'rb'))
bu_grad, bi_grad, p_grad, q_grad, t_grad, (beta1_grad, beta2_grad, beta3_grad) = check_ordmm_gradient.check_ordmm_gradient(
                                                                                                model_ordmm,
                                                                                                convert_input(model_ordmm,
                                                                                                              train_100k),
                                                                                                 reg_dict)
abs_p, abs_q, abs_bu, abs_bi = abs(p_grad), abs(q_grad), abs(bu_grad), abs(bi_grad)
ordmm_grad = [np.max(abs_p), np.min(abs_p), np.mean(p_grad),
              np.max(abs_q), np.min(abs_q), np.mean(q_grad),
              np.max(abs_bu), np.min(abs_bu), np.mean(bu_grad),
              np.max(abs_bi), np.min(abs_bi), np.mean(bi_grad),
              '', '', '',
              t_grad[0], beta1_grad, beta2_grad, beta3_grad]

################# Ord ###########################

reg_dict = {'reg_bu': 0.001,
            'reg_bi': 0.001,
            'reg_pu': 0.001,
            'reg_qi': 0.001,
            'reg_xi': 0.001,
            'reg_t': 0.001,
            'reg_beta': 0.001}
model_ord = pickle.load(open('models\model_ord_ml100k_f%d.pkl'%dim, 'rb'))
bu_grad, bi_grad, p_grad, q_grad, t_grad, x_grad, (beta1_grad, beta2_grad, beta3_grad) = check_ord_gradient.check_ord_gradient(
                                                                                                    model_ord,
                                                                                                    convert_input(model_ord,
                                                                                                                  train_100k),
                                                                                                    reg_dict)
abs_p, abs_q, abs_bu, abs_bi, abs_x = abs(p_grad), abs(q_grad), abs(bu_grad), abs(bi_grad), abs(x_grad) 
ord_grad = [np.max(abs_p), np.min(abs_p), np.mean(p_grad),
            np.max(abs_q), np.min(abs_q), np.mean(q_grad),
            np.max(abs_bu), np.min(abs_bu), np.mean(bu_grad),
            np.max(abs_bi), np.min(abs_bi), np.mean(bi_grad),
            np.max(abs_x), np.min(abs_x), np.mean(x_grad),
            t_grad[0], beta1_grad, beta2_grad, beta3_grad]

gradient_df = pd.DataFrame()
gradient_df['paramters'] = ['Max |P|', 'Min |P|', 'Mean P',
                            'Max |Q|', 'Min |Q|', 'Mean Q',
                            'Max |bu|', 'Min |bu|', 'Mean bu',
                            'Max |bi|', 'Min |bi|', 'Mean bi',
                            'Max |x|', 'Min |x|', 'Mean x',
                            't1', 'beta1', 'beta2', 'beta3']
gradient_df['SVD based model'] = svd_grad
gradient_df['SVD++ model'] = svdpp_grad
gradient_df['SVD based OrdRec'] = ordmm_grad
gradient_df['SVD++ based OrdRec'] = ord_grad

gradient_df.to_csv('tables\ML_100k_grad_f%d(table8).csv'%dim, index=False, float_format='%.6f')

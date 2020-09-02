import os
import sys
import numpy as np
import pandas as pd
import pickle
package_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(package_path)
from Ting_MF import ting_svd, ting_svdpp, ting_ord, ting_ordmm, my_metric

def convert_input(model, df):
    data = df.copy()
    data.loc[:, 'user'] = data.user.map(model.user2idx)
    data.loc[:, 'item'] = data.item.map(model.item2idx)
    
    return data

train_100k = pd.read_csv(r'ml-100k\train_100k.csv')
test_100k = pd.read_csv(r'ml-100k\test_100k.csv')

def fit_with_each_model(dim):
    
    ################ SVD   ######################
    
    lr_rate =  2.5
    weight_decay= 0.0004
    batch_size = 500
    epoches = 20
    model_svd = ting_svd.SVD(train_100k, dim)
    model_svd.train(lr_rate, weight_decay, batch_size, epoches, verbose=False)
    pickle.dump(model_svd, open('models\model_svd_ml100k_f%d.pkl'%dim, 'wb'))
    
    svd_rmse = model_svd.evaluate(test_100k)
    svd_pre = my_metric.predict_svd(model_svd, convert_input(model_svd, test_100k))
    svd_fcp = my_metric.FCP1(test_100k, svd_pre)
    
    ################ SVDpp   ####################
    
    lr_rate =  2.5
    weight_decay= 0.0004
    batch_size = 500
    epoches = 20
    model_svdpp = ting_svdpp.SVDpp(train_100k, dim)
    model_svdpp.train(lr_rate, weight_decay, batch_size, epoches, verbose=False)
    pickle.dump(model_svdpp, open('models\model_svdpp_ml100k_f%d.pkl'%dim, 'wb'))
    
    svdpp_rmse = model_svdpp.evaluate(test_100k)
    svdpp_pre = my_metric.predict_svd(model_svdpp, convert_input(model_svdpp, test_100k))
    svdpp_fcp = my_metric.FCP1(test_100k, svdpp_pre)
      
    ################ Ordmm   ####################
    
    batch_size = 500
    lr_all = 1.6
    reg_all = 0.001
    lr_bu, lr_bi, lr_pu, lr_qi, lr_t, lr_beta = 2.605, 2.605, 2.474, 2.474, 1.816, np.array([1.816, 1.816, 1.684])  
    reg_bu, reg_bi, reg_pu, reg_qi, reg_t, reg_beta = 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001
    epoches = 20

    model_ordmm = ting_ordmm.ordmm_rec(train_100k, dim)
    model_ordmm.train(epoches=epoches, batch_size=batch_size, lr_all=lr_all, reg_all=reg_all,
                  lr_bu=lr_bu, lr_bi=lr_bi, lr_pu=lr_pu, lr_qi=lr_qi, lr_t=lr_t, lr_beta=lr_beta,  
                  reg_bu=reg_bu, reg_bi=reg_bi, reg_pu=reg_pu, reg_qi=reg_qi, reg_t=reg_t, reg_beta=reg_beta,
                  verbose=False)
    pickle.dump(model_ordmm, open('models\model_ordmm_ml100k_f%d.pkl'%dim, 'wb'))
    
    
    ordmm_rmse = model_ordmm.evaluate(test_100k)
    ordmm_pre = my_metric.predict_ord(model_ordmm, convert_input(model_ordmm, test_100k))
    ordmm_fcp = my_metric.FCP1(test_100k, ordmm_pre)
    
    ################ Ord   ######################
    
    batch_size = 500
    lr_all = 1.6
    reg_all = 0.001
    lr_bu, lr_bi, lr_pu, lr_qi, lr_xi, lr_t, lr_beta = 2.605, 2.605, 2.474, 2.474, 2.5, 1.816, np.array([1.816, 1.816, 1.684])  
    reg_bu, reg_bi, reg_pu, reg_qi, reg_xi, reg_t, reg_beta = 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001
    epoches = 20

    model_ord = ting_ord.ord_rec(train_100k, dim)
    model_ord.train(epoches=epoches, batch_size=batch_size, lr_all=lr_all, reg_all=reg_all,
                  lr_bu=lr_bu, lr_bi=lr_bi, lr_pu=lr_pu, lr_qi=lr_qi, lr_xi=lr_xi, lr_t=lr_t, lr_beta=lr_beta,  
                  reg_bu=reg_bu, reg_bi=reg_bi, reg_pu=reg_pu, reg_qi=reg_qi, reg_xi=reg_xi, reg_t=reg_t, reg_beta=reg_beta,
                  verbose=False)
    pickle.dump(model_ord, open('models\model_ord_ml100k_f%d.pkl'%dim, 'wb'))
    
    
    ord_rmse = model_ord.evaluate(test_100k)
    ord_pre = my_metric.predict_ord(model_ord, convert_input(model_ord, test_100k))
    ord_fcp = my_metric.FCP1(test_100k, ord_pre)
    
    return [svd_rmse, svdpp_rmse, ordmm_rmse, ord_rmse], [svd_fcp, svdpp_fcp, ordmm_fcp, ord_fcp]

RMSE_table = pd.DataFrame({'Method': ['SVD based model', 'SVD++ model', 'SVD based OrdRec', 'SVD++ based OrdRec']})
FCP_table = pd.DataFrame({'Method': ['SVD based model', 'SVD++ model', 'SVD based OrdRec', 'SVD++ based OrdRec']})

for dim in [50, 100, 200]:
    rmses, fcps = fit_with_each_model(dim)
    RMSE_table['f=%d'%dim] = rmses
    FCP_table['f=%d'%dim] = fcps

RMSE_table.to_csv('tables\MovieLens_100k_RMSE_table(table5).csv', index=False)
FCP_table.to_csv('tables\MovieLens_100k_FCP_table(table5).csv', index=False)

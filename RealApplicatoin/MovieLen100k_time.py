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

def fit_with_each_model(dim):
    
    epoches = 1
    ################ SVD   ######################
    
    lr_rate =  2.5
    weight_decay= 0.0004
    batch_size = 500
    
    model_svd = ting_svd.SVD(train_100k, dim)
    svd_time = model_svd.train(lr_rate, weight_decay, batch_size, epoches, verbose=False)
    
    ################ SVDpp   ####################
    
    lr_rate =  2.5
    weight_decay= 0.0004
    batch_size = 500
   
    model_svdpp = ting_svdpp.SVDpp(train_100k, dim)
    svdpp_time = model_svdpp.train(lr_rate, weight_decay, batch_size, epoches, verbose=False)
     
    ################ Ordmm   ####################
    
    batch_size = 500
    lr_all = 1.6
    reg_all = 0.001
    lr_bu, lr_bi, lr_pu, lr_qi, lr_t, lr_beta = 2.605, 2.605, 2.474, 2.474, 1.816, np.array([1.816, 1.816, 1.684])  
    reg_bu, reg_bi, reg_pu, reg_qi, reg_t, reg_beta = 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001

    model_ordmm = ting_ordmm.ordmm_rec(train_100k, dim)
    ordmm_time = model_ordmm.train(epoches=epoches, batch_size=batch_size, lr_all=lr_all, reg_all=reg_all,
                  lr_bu=lr_bu, lr_bi=lr_bi, lr_pu=lr_pu, lr_qi=lr_qi, lr_t=lr_t, lr_beta=lr_beta,  
                  reg_bu=reg_bu, reg_bi=reg_bi, reg_pu=reg_pu, reg_qi=reg_qi, reg_t=reg_t, reg_beta=reg_beta,
                  verbose=False)

    ################ Ord   ######################
    
    batch_size = 500
    lr_all = 1.6
    reg_all = 0.001
    lr_bu, lr_bi, lr_pu, lr_qi, lr_xi, lr_t, lr_beta = 2.605, 2.605, 2.474, 2.474, 2.5, 1.816, np.array([1.816, 1.816, 1.684])  
    reg_bu, reg_bi, reg_pu, reg_qi, reg_xi, reg_t, reg_beta = 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001

    model_ord = ting_ord.ord_rec(train_100k, dim)
    ord_time = model_ord.train(epoches=epoches, batch_size=batch_size, lr_all=lr_all, reg_all=reg_all,
                  lr_bu=lr_bu, lr_bi=lr_bi, lr_pu=lr_pu, lr_qi=lr_qi, lr_xi=lr_xi, lr_t=lr_t, lr_beta=lr_beta,  
                  reg_bu=reg_bu, reg_bi=reg_bi, reg_pu=reg_pu, reg_qi=reg_qi, reg_xi=reg_xi, reg_t=reg_t, reg_beta=reg_beta,
                  verbose=False)
    
    return [svd_time, svdpp_time, ordmm_time, ord_time]

TIME_table = pd.DataFrame({'Method': ['SVD based model', 'SVD++ model', 'SVD based OrdRec', 'SVD++ based OrdRec']})

for dim in [50, 100, 200]:
    times= fit_with_each_model(dim)
    TIME_table['f=%d'%dim] = times

TIME_table.to_csv('tables\MovieLens_100k_TimeCost(table7).csv', index=False)

import numpy as np
import pandas as pd
from tqdm.auto import  tqdm

import os
import sys
package_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(package_path)
from Ting_MF import ting_ordmm

def SimulationOrdRec(num_user, num_item, dim, N_list):
    
    # initial parameters
    r = 5
    P = np.random.normal(scale=0.1, size=(dim, num_user))
    Q = np.random.normal(scale=0.1, size=(dim, num_item))
    bu = np.random.normal(size=num_user)
    bi = np.random.normal(size=num_item)
    t = np.random.randn(1)
    beta = np.random.randn(r - 2)
    
    # compute cut-points
    threshold = np.append(t, np.exp(beta)).cumsum()
    threshold = np.concatenate([[-1*np.inf], threshold, [np.inf]])
    
    # let y_{ui} = b_u + b_i + P^TQ
    # P(\widehat r_{ui} \leq r) = \frac{1}{1 + e^{y_{ui} - t_r}}
    y_ui = bu[:, np.newaxis] + bi[np.newaxis, :] + P.T.dot(Q)
    cdfs = 1 / (1+np.exp(y_ui[:, :, np.newaxis] - threshold[np.newaxis, np.newaxis, :]))
    pdfs = cdfs[:, :, 1:] - cdfs[:, :, :-1]
    
    # make data df
    data = pd.DataFrame(columns=['user', 'item', 'rating'], index=range(num_user*num_item))
    # np.repeat repeats like [0,0,1,1,2,2]
    data.loc[:, 'user'] = np.repeat(range(num_user), num_item)
    # np.tile repeats like [0,1,2,0,1,2]
    data.loc[:, 'item'] = np.tile(range(num_item), num_user)
    data.loc[:, 'rating'] = np.apply_along_axis(func1d=lambda x: np.asscalar(np.random.choice(range(r), size=1, p=x)),
                                                arr=pdfs.reshape(-1, r), axis=1) + 1
    
    # store RMSE of each paramters trained by model and real parameters
    f = []
    # train OrdRec when dimensionality=dim, n_user=num_user, n_item=num_item, num of sampled ratings=N 
    for N in N_list:
        train = data.sample(n=N).reset_index(drop=True)
        batch_size = 500
        lr_all = 1.6
        reg_all = 0.001
        lr_bu, lr_bi, lr_pu, lr_qi, lr_t, lr_beta = 2.605, 2.605, 2.474, 2.474, 1.816, np.array([1.816, 1.816, 1.684])  
        reg_bu, reg_bi, reg_pu, reg_qi, reg_t, reg_beta = 0., 0., 0., 0., 0., 0.
        epoches = 60

        ######################################
        print('Now fitting model with num_user={0}, num_item={1}, f={2}, N={3}'.\
              format(num_user, num_item, dim, N))
        model_ordmm = ting_ordmm.ordmm_rec(train, dim)
        model_ordmm.train(epoches=epoches, batch_size=batch_size, lr_all=lr_all, reg_all=reg_all,
                      lr_bu=lr_bu, lr_bi=lr_bi, lr_pu=lr_pu, lr_qi=lr_qi, lr_t=lr_t, lr_beta=lr_beta,  
                      reg_bu=reg_bu, reg_bi=reg_bi, reg_pu=reg_pu, reg_qi=reg_qi, reg_t=reg_t, reg_beta=reg_beta, verbose=False)
        
        # compare RMSE of parametes trained by model and real parameters
        # RMSE of P
        f.append(np.sqrt(np.mean((model_ordmm.P - P)**2))) 
        # RMSE of Q
        f.append(np.sqrt(np.mean((model_ordmm.Q - Q)**2))) 
        #RMSE of b_u
        f.append(np.sqrt(np.mean((model_ordmm.b_u - bu)**2)))
        #RMSE of b_i
        f.append(np.sqrt(np.mean((model_ordmm.b_i - bi)**2)))
        #RMSE of t
        f.append(np.sqrt((model_ordmm.t - t)**2)[0])
        #RMSE of beta
        f.append(np.sqrt(np.mean((beta - model_ordmm.beta)**2)))

    return f

num_user_item_Ns = [(100, 500, [30000, 50000]), (900, 1600, [50000, 80000])]
dimentions = [20, 50, 100]

for (num_user, num_item, N_list) in num_user_item_Ns:
    
    results = pd.DataFrame()
    results['N'] = np.repeat(N_list, 6)
    results['parameters'] = np.tile(['P', 'Q', 'b_u', 'b_i', 't', 'beta'], 2)
    
    for dim in dimentions:
        
        results['f=%d'%dim] = SimulationOrdRec(num_user, num_item, dim, N_list)
        
    
    print('Save output table when num_user={0}, num_item={1} as M_{2}_N_{3}.csv'.format(num_user, num_item,num_user, num_item))
    results.to_csv('M_{0}_N_{1}.csv'.format(num_user, num_item), index=False)

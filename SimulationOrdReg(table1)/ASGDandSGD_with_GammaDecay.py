from OptimizationMethods import SGD_GammaDecay
import time
import pickle
import numpy as np
import pandas as pd
from tqdm.auto import  tqdm


lr_all = 1.6
reg_all = 0.001
lr_pu, lr_t, lr_beta = 2.1, 2.8, np.array([1.11, 1.052, 1.05])  
reg_pu, reg_t, reg_beta = 0., 0., 0.
epoches = 1

times = []
rmse_p = []
rmse_t = []
rmse_beta = []

rmse_avg_p = []
rmse_avg_t = []
rmse_avg_beta = []

for i in tqdm(range(20), ascii=True):
    
    data = pd.read_csv('simulation_data\data_%d.csv'%i)
    params = pickle.load(open('simulation_data\parameters_%d.pkl'%i, 'rb'))
    P = params['P']
    t = params['t']
    beta = params['beta']
    
    model = SGD_GammaDecay.ord_reg(data)
    start = time.clock()
    model.train(epoches, 
                lr_all, reg_all,
                lr_pu, lr_t, lr_beta,  
                reg_pu, reg_t, reg_beta)
    times.append(time.clock() - start)
    
    P_avg = np.mean(model.pu_list, axis=0)
    t_avg = np.mean(model.t_list, axis=0)
    beta_avg = np.mean(model.beta_list, axis=0)
    
    rmse_p.append(np.sqrt(np.mean((model.P - P)**2)))
    rmse_t.append(np.sqrt(np.mean((model.t - t)**2)))
    rmse_beta.append(np.sqrt(np.mean((model.beta - beta)**2)))
     
    rmse_avg_p.append(np.sqrt(np.mean((P_avg - P)**2)))
    rmse_avg_t.append(np.sqrt(np.mean((t_avg - t)**2)))
    rmse_avg_beta.append(np.sqrt(np.mean((beta_avg - beta)**2)))


# print('=======SGD with gamma_n decayed=======')
# print('time cost: \n ==> {:.4f} seconds'.format(np.mean(times)))
# print('RMSE of P: \n ==> {:.4f}'.format(np.mean(rmse_p)))
# print('RMSE of t: \n ==> {:.4f}'.format(np.mean(rmse_t)))
# print('RMSE of beta: \n ==> {:.4f}'.format(np.mean(rmse_beta)))
# print('RMSE of overall: \n ==> {:.4f}'.format(np.mean([np.mean(rmse_p),
#                                                        np.mean(rmse_t),
#                                                        np.mean(rmse_beta)])))

# print('=======ASGD with gamma_n decayed=======')
# print('time cost: \n ==> {:.4f} seconds'.format(np.mean(times)))
# print('RMSE of P: \n ==> {:.4f}'.format(np.mean(rmse_avg_p)))
# print('RMSE of t: \n ==> {:.4f}'.format(np.mean(rmse_avg_t)))
# print('RMSE of beta: \n ==> {:.4f}'.format(np.mean(rmse_avg_beta)))
# print('RMSE of overall: \n ==> {:.4f}'.format(np.mean([np.mean(rmse_avg_p),
#                                                        np.mean(rmse_avg_t),
#                                                        np.mean(rmse_avg_beta)])))



df = pd.DataFrame(index=['SGD with gamma_n decayed', 'ASGD with gamma_n decayed'], columns=['w', 't', 'beta', 'overall', 'training time'])
df.loc['SGD with gamma_n decayed',:] = [np.mean(rmse_p),
                                        np.mean(rmse_t),
                                        np.mean(rmse_beta),
                                        np.mean([np.mean(rmse_p),
                                                 np.mean(rmse_t),
                                                 np.mean(rmse_beta)]),
                                        np.mean(times)]

df.loc['ASGD with gamma_n decayed',:] = [np.mean(rmse_avg_p),
                                        np.mean(rmse_avg_t),
                                        np.mean(rmse_avg_beta),
                                        np.mean([np.mean(rmse_avg_p),
                                                 np.mean(rmse_avg_t),
                                                 np.mean(rmse_avg_beta)]),
                                        np.mean(times)]


df.to_csv('ToConcatenate\ASGDandSGD_with_GammaDecay_df.csv')













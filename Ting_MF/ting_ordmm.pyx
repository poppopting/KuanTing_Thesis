  # noqa
import random
import numpy as np
cimport numpy as np
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()
#from tqdm import tqdm_notebook
import time
import gc
import numpy_indexed as npi
class ordmm_rec():

    # Initializing the user-movie rating matrix, no. of latent features
    def __init__(self, data, dim):
        tmp_data = data.copy()
        user_pool = np.unique(data.iloc[:, 0])
        item_pool = np.unique(data.iloc[:, 1])
        
        self.num_users, self.num_items = len(user_pool), len(item_pool)
        self.user2idx = dict(zip(user_pool, range(1 ,self.num_users+1)))
        self.item2idx = dict(zip(item_pool, range(1, self.num_items+1)))
        
        tmp_data.loc[:, 'user'] = data.iloc[:, 0].map(self.user2idx)
        tmp_data.loc[:, 'item'] = data.iloc[:, 1].map(self.item2idx)
        self.dim = dim
        
        self.len_ratings = len(data)

        self.ratings = tmp_data.values.astype('uint16')
       
        del tmp_data
        gc.collect()     
        
        cdef np.ndarray[np.double_t] b_u, b_i, t
        cdef np.ndarray[np.double_t, ndim=1] beta
        cdef np.ndarray[np.double_t, ndim=2] P, Q
                
        P = np.random.normal(scale=0.1, size=(self.dim, self.num_users))
        Q = np.random.normal(scale=0.1, size=(self.dim, self.num_items))

        # Initializing the bias terms
        b_u = np.zeros(self.num_users)
        b_i = np.zeros(self.num_items)

        t = np.random.randn(1)
        beta = np.random.randn(3)
        
        self.P = P
        self.Q = Q

        # Initializing the bias terms
        self.b_u = b_u 
        self.b_i = b_i 

        self.t = t
        self.beta = beta
    
    def thresholds_all(self):
        cdef np.ndarray[np.double_t] thresholds, thresholds_all 
        thresholds = np.cumsum(np.concatenate([self.t, np.exp(self.beta)]))
        thresholds_all = np.concatenate([[-1*np.inf], thresholds, [np.inf]])
        return thresholds_all
    # Initializing user-feature and movie-feature matrix 
    def train(self,
              epoches, batch_size, lr_all=.005, reg_all=.02,
              lr_bu=None, lr_bi=None, lr_pu=None, lr_qi=None, lr_t=None, lr_beta=None,  
              reg_bu=None, reg_bi=None, reg_pu=None, reg_qi=None, reg_t=None, reg_beta=None, verbose=True
             ):
        self.batch_size = batch_size
        # learning rate
        self.lr_bu = lr_bu if lr_bu is not None else lr_all
        self.lr_bi = lr_bi if lr_bi is not None else lr_all
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_qi = lr_qi if lr_qi is not None else lr_all

        self.lr_t = lr_t if lr_t is not None else lr_all
        self.lr_beta = lr_beta if lr_beta is not None else lr_all
        # regularization
        self.reg_bu = reg_bu if reg_bu is not None else reg_all
        self.reg_bi = reg_bi if reg_bi is not None else reg_all
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_qi = reg_qi if reg_qi is not None else reg_all

        self.reg_t = reg_t if reg_t is not None else reg_all
        self.reg_beta = reg_beta if reg_beta is not None else reg_all
        
        cdef double start, log_err, rmse_err
        cdef int i, ix
        cdef np.ndarray[np.uint16_t, ndim=1] user_s, item_s, ratings
        cdef np.ndarray[np.double_t, ndim=2] cdfs, score_dist
        cdef np.ndarray[np.double_t, ndim=1] prediction 
        print('Begin training')
        start = time.clock()
        
        for i in tqdm(range(epoches), ascii=True):
            np.random.shuffle(self.ratings)
            
            for ix in range(0, len(self.ratings), batch_size):
   
                user_s, item_s, ratings = self.ratings[ix: ix+batch_size].T           
                cdfs, score_dist, prediction = self.get_rating(user_s, item_s)
                self.sgd(user_s, item_s, ratings, cdfs, prediction)

            log_err, rmse_err = self.rmse()
            if verbose:
                print('epoch : {0:2d} / {1} Log-likelihood = {2:.4f}, RMSE = {3:.4f}'.\
                  format(i+1, epoches, log_err, rmse_err))
        print('Training complete \nTime past : %.2f seconds'%(time.clock()-start))
        return time.clock()-start
    def evaluate(self, test):
        
        test_tmp = test.copy()
        test_tmp.loc[:, 'user'] = test.iloc[:, 0].map(self.user2idx)
        test_tmp.loc[:, 'item'] = test.iloc[:, 1].map(self.item2idx)
        test_data = test_tmp.values.astype('uint16')
        error = 0
        for i in range(0, len(test_data), 500):
            user_s, item_s, ratings = test_data[i:i+500].T
            pre_score = self.get_rating(user_s, item_s)[1].argmax(axis=1) + 1
            error += np.sum(pow(np.asarray(ratings) - pre_score, 2))
           
        print('RMSE of testing data : %.4f'%np.sqrt(error/len(test_data))) 
        
        return np.sqrt(error/len(test_data))

    # Computing total mean squared error
    def rmse(self):

        cdef double log_error = 0, rmse_error = 0
        cdef int i
        cdef np.ndarray[np.double_t, ndim=1] pre_score_dist
        cdef np.ndarray[np.uint16_t, ndim=1] user_s, item_s, ratings
        cdef np.ndarray[np.intp_t, ndim=1] pre_score
        
        for i in range(0, self.len_ratings, 500):
            user_s, item_s, ratings = self.ratings[i:i+500].T
            
            pre_score_dist = self.get_rating(user_s, item_s)[1][np.arange(len(user_s)), ratings-1]
            log_error += np.sum(np.log(pre_score_dist))
            
            pre_score = self.get_rating(user_s, item_s)[1].argmax(axis=1) + 1
            rmse_error += np.sum(pow(ratings - pre_score, 2))
        
        log_error /= self.len_ratings
        log_error -= self.reg_bu*np.sum(self.b_u**2)/2 + self.reg_bi*np.sum(self.b_i**2)/2 + \
                     self.reg_pu*np.sum(self.P**2)/2 + self.reg_qi*np.sum(self.Q**2)/2 + \
                     self.reg_t*np.sum(self.t**2)/2 + self.reg_beta*np.sum(self.beta**2)/2
        return log_error, np.sqrt(rmse_error/self.len_ratings)
    
    # Stochastic gradient descent to get optimized P and Q matrix
    def sgd(self, i_s, j_s, trues, cdfs_s, prediction):
        self.trues = trues
        self.cdfs_s = cdfs_s
        
        cdef int batch
        cdef double t, beta1, beta2, beta3
        cdef double update_beta1, update_beta2, update_beta3
        cdef np.ndarray[np.double_t, ndim=1] bigger_r, smaller_r, diff
        cdef np.ndarray[np.uint16_t, ndim=1] unique_i, unique_j
        cdef np.ndarray[np.double_t, ndim=1] update_bu, update_bi, update_beta
        cdef np.ndarray[np.double_t, ndim=2] tmp_p, tmp_q, update_p, update_q
        batch = len(trues)
        bigger_r = 1 - cdfs_s[np.arange(batch), trues]
        smaller_r = cdfs_s[np.arange(batch), trues-1]
        diff = bigger_r - smaller_r 
        # update bu
        unique_i, update_bu = npi.group_by(i_s).sum(-diff, axis=0)
        self.b_u[unique_i-1] += self.lr_bu * (update_bu / batch - self.reg_bu*self.b_u[unique_i-1])
        # update bi
        unique_j, update_bi = npi.group_by(j_s).sum(-diff, axis=0)
        self.b_i[unique_j-1] += self.lr_bi * (update_bi / batch - self.reg_bi*self.b_i[unique_j-1])
        # update pu
        tmp_p = -self.Q[:, j_s-1] * diff
        unique_i, update_p = npi.group_by(i_s).sum(tmp_p, axis=1)
        self.P[:, unique_i-1] += self.lr_pu * (update_p / batch - self.reg_pu*self.P[:, unique_i-1])
        # update qi
        tmp_q = -self.P[:, i_s-1] * diff
        unique_j, update_q = npi.group_by(j_s).sum(tmp_q, axis=1)
        self.Q[:, unique_j-1] += self.lr_qi * (update_q / batch - self.reg_qi*self.Q[:, unique_j-1])
        # update t
        t = self.t
        self.t += self.lr_t * (np.sum(diff) / batch - self.reg_t*self.t)

        beta1, beta2, beta3 = self.beta.copy()
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

        update_beta = np.asarray([update_beta1, update_beta2, update_beta3])
        update_beta /= batch
        self.beta += self.lr_beta * (update_beta - self.reg_beta * self.beta)

        return self
    
    # Ratings for user i and moive j
    def get_rating(self, i_s, j_s):
        cdef np.ndarray[np.double_t, ndim=1] prediction
        cdef np.ndarray[np.double_t, ndim=2] cdfs, score_dist

        prediction = self.b_u[i_s-1] + self.b_i[j_s-1] + np.diag(self.P[:, i_s-1].T.dot(self.Q[:, j_s-1]))
        cdfs = 1/(1+np.exp(prediction[:, np.newaxis] - self.thresholds_all()[np.newaxis, :]))
        score_dist = cdfs[:, 1:] - cdfs[:, :-1] 

        return cdfs, score_dist, prediction
        
    # Full user-movie rating matrix
    def full_matrix(self):

        svd_pp_score =  self.b_u[:,np.newaxis] + self.b_i[np.newaxis:,] + self.P.T.dot(self.Q)      
        ord_tmp = 1/(1+np.exp(svd_pp_score[:, :, np.newaxis] - self.thresholds_all()[np.newaxis, np.newaxis, :]))
        ord_score = (ord_tmp[:, :, 1:] - ord_tmp[:, :, :-1]).argmax(axis=2) + 1
        return ord_score, ord_tmp, svd_pp_score

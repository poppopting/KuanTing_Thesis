  # noqa
import numpy as np
cimport numpy as np
cimport cython
from operator import itemgetter
from tqdm.auto import tqdm
import time
import numpy_indexed as npi
import random
import gc
class SVD():

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

        self.mu = np.mean(data.iloc[:, 2].tolist())
        self.len_ratings = len(data)

        self.ratings = tmp_data.values.astype('uint16')
       
        del tmp_data
        gc.collect()
    # Initializing user-feature and movie-feature matrix 
    def train(self, lr_rate, weight_decay, batch_size, epoches, verbose=True):
        
        cdef double start
        cdef int i, user, item, ix
        cdef list batches
        cdef np.ndarray[np.double_t] predict

        cdef double curren_loss

        
        cdef np.ndarray[np.double_t] b_u, b_i
        cdef np.ndarray[np.double_t, ndim=2] P, Q, batch

    
        P = np.random.normal(scale=0.1, size=(self.num_users, self.dim))
        Q = np.random.normal(scale=0.1, size=(self.num_items, self.dim))

        # Initializing the bias terms
        b_u = np.zeros(self.num_users)
        b_i = np.zeros(self.num_items)
        
        self.b_u = b_u
        self.b_i = b_i
        self.P = P
        self.Q = Q
    
        self.lr_rate, self.weight_decay = lr_rate, weight_decay
        print('Begin training')
        start = time.clock()

        for i in tqdm(range(epoches), ascii=True):
            np.random.shuffle(self.ratings)
            
            for ix in range(0, len(self.ratings), batch_size):
    
                user_s, item_s, ratings = self.ratings[ix:ix+batch_size].T
                
                predict = self.get_rating(user_s, item_s)
                
                self.sgd(user_s, item_s, ratings, predict, lr_rate, weight_decay)
                
            
            obj, metric = self.rmse()
            if verbose:
                print('epoch : {0:2d} / {1} Reg_MSE = {2:.4f}, RMSE = {3:.4f}'.\
                      format(i+1, epoches, obj, metric))
           
           # self.evaluate(test)
        print('Training complete \n\nTime past : %.2f seconds'%(time.clock()-start))   
        return time.clock()-start
    def evaluate(self, test):
        
        test_tmp = test.copy()
        test_tmp.loc[:, 'user'] = test.iloc[:, 0].map(self.user2idx)
        test_tmp.loc[:, 'item'] = test.iloc[:, 1].map(self.item2idx)
        cdef np.ndarray[np.double_t] predict #, user_s, item_s, ratings
        cdef double error=0
        cdef int ix
        cdef np.ndarray[np.uint16_t, ndim=2] test_data = test_tmp.values.astype('uint16')    
        
        for ix in range(0, len(test_data), 500):
            user_s, item_s, ratings = test_data[ix:ix+500].T
           
            predict = self.get_rating(user_s, item_s)
            error += np.sum(pow(ratings - predict, 2))
            
        print('RMSE of testing data : %.4f'%np.sqrt(error/len(test_data))) 

        return np.sqrt(error/len(test_data))

    # Computing total mean squared error
    def rmse(self):
           
        cdef np.ndarray[np.double_t] predict
        cdef double error=0
        cdef int ix
        
        for ix in range(0, len(self.ratings), 500):
            user_s, item_s, ratings = self.ratings[ix:ix+500].T
           
            predict = self.get_rating(user_s, item_s)
            error += np.sum(pow(ratings - predict, 2))
            
        rmse = np.sqrt(error / self.len_ratings)
        reg_mse = error / (2 * self.len_ratings)
        reg_mse += self.weight_decay*(np.sum(self.b_u**2)/2 + np.sum(self.b_i**2)/2 + \
                                      np.sum(self.P**2)/2 + np.sum(self.Q**2)/2 )
        
        return reg_mse, rmse

    # Stochastic gradient descent to get optimized P and Q matrix
    def sgd(self, i_s, j_s, trues, predicts, lr_rate, weight_decay):
        
        cdef np.ndarray[np.double_t]  e_s, b_u = self.b_u, b_i = self.b_i, update_bu, update_bi
        cdef np.ndarray[np.uint16_t]  unique_i, unique_j
        cdef np.ndarray[np.double_t, ndim=2] P = self.P, Q = self.Q, tmp_q, tmp_p
        cdef np.ndarray[np.double_t, ndim=2] update_P, update_Q
        
        e_s = (trues - predicts)
        batch = len(e_s)
        
        tmp_q = e_s[:, np.newaxis] * P[i_s-1, :] 
        tmp_p = e_s[:, np.newaxis] * Q[j_s-1, :] 

        unique_i, update_bu = npi.group_by(i_s).sum(e_s, axis=0)
        b_u[unique_i-1] += lr_rate * (update_bu / batch - weight_decay * b_u[unique_i-1])
       
        unique_j, update_bi = npi.group_by(j_s).sum(e_s, axis=0)
        b_i[unique_j-1] += lr_rate * (update_bi / batch - weight_decay * b_i[unique_j-1])

        unique_i, update_P = npi.group_by(i_s).sum(tmp_p, axis=0)
        P[unique_i-1,:] += lr_rate * (update_P / batch - weight_decay * P[unique_i-1,:])  
   
        unique_j, update_Q = npi.group_by(j_s).sum(tmp_q, axis=0)
  
        Q[unique_j-1,:] += lr_rate * (update_Q / batch - weight_decay * Q[unique_j-1,:])
        
        self.b_u = b_u
        self.b_i = b_i
        self.P = P
        self.Q = Q
        
   
        return self

    # Ratings for user i and moive j
    def get_rating(self, i_s, j_s):    
        cdef np.ndarray[np.double_t]  prediction
        cdef np.ndarray[np.double_t] b_u = self.b_u
        cdef np.ndarray[np.double_t] b_i = self.b_i
        cdef np.ndarray[np.double_t, ndim=2] P = self.P
        cdef np.ndarray[np.double_t, ndim=2] Q = self.Q
        
        prediction = self.mu + b_u[i_s-1] + b_i[j_s-1] + np.diag(P[i_s-1,:].dot(Q[j_s-1,:].T))

        return prediction

    # Full user-movie rating matrix
    def full_matrix(self):
        cdef np.ndarray[np.double_t] b_u = self.b_u
        cdef np.ndarray[np.double_t] b_i = self.b_i
        cdef np.ndarray[np.double_t, ndim=2] P = self.P
        cdef np.ndarray[np.double_t, ndim=2] Q = self.Q
        
        return self.mu + b_u[:,np.newaxis] + b_i[np.newaxis,:] + P.dot(Q.T)
        

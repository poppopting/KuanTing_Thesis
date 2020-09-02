import numpy as np
import pandas as pd
import time
from tqdm.auto import  tqdm
import numpy_indexed as npi


class ord_reg():

    # Initializing the user-movie rating matrix, no. of latent features
    def __init__(self, data):
        
        self.N, self.dim = data.iloc[:,:-1].shape
        self.data = data.copy().values
        
        self.P = np.random.normal(scale=0.1, size=(self.dim))
        self.t = np.random.randn(1)
        self.beta = np.random.randn(3)
        
        
        self.pu_list = []
        self.t_list = []
        self.beta_list = []
        
    def thresholds_all(self):

        thresholds = np.cumsum(np.concatenate([self.t, np.exp(self.beta)]))
        thresholds_all = np.concatenate([[-1*np.inf], thresholds, [np.inf]])

        return thresholds_all
    # Initializing user-feature and movie-feature matrix 
    def train(self,
              epoches, batch_size, lr_all=.005, reg_all=.02,
              lr_pu=None, lr_t=None, lr_beta=None,  
              reg_pu=None, reg_t=None, reg_beta=None):
        
        self.batch_size = batch_size
        # learning rate
        self.lr_pu = lr_pu if lr_pu is not None else lr_all
        self.lr_t = lr_t if lr_t is not None else lr_all
        self.lr_beta = lr_beta if lr_beta is not None else lr_all
        
        # regularization
        self.reg_pu = reg_pu if reg_pu is not None else reg_all
        self.reg_t = reg_t if reg_t is not None else reg_all
        self.reg_beta = reg_beta if reg_beta is not None else reg_all
        
       # print('Begin training')
        start = time.clock()
        alpha = 0.5
        for i in range(epoches):
            
            np.random.shuffle(self.data)
            n = 1
            for ix in range(0, self.N, batch_size):
                
                x_arr = self.data[ix: ix+batch_size][:, :-1]
                rating = self.data[ix: ix+batch_size][:, -1].astype(int)
                
                cdfs, score_dist, prediction = self.get_rating(x_arr)
                
                self.lr_pu = lr_pu * pow(n, -alpha)
                self.lr_t = lr_t * pow(n, -alpha)
                self.lr_beta = lr_beta * pow(n, -alpha)
                
                self.sgd(x_arr, rating, cdfs, prediction)
                n += 1
                
                self.pu_list.append(self.P.copy())
                self.t_list.append(self.t.copy())
                self.beta_list.append(self.beta.copy())
            log_err, rmse_err = self.rmse()

        
    def evaluate(self, test_data):
    
        error = 0
        
        for i in range(0, len(test_data), 500):
            user_s, item_s, ratings = zip(*test_data[i:i+500] )
            pre_score = self.get_rating(np.asarray(user_s), np.asarray(item_s))[1].argmax(axis=1) + 1
            error += np.sum(pow(np.asarray(ratings) - pre_score, 2))
           
        print('RMSE of testing data : %.4f'%np.sqrt(error/len(test_data))) 

    # Computing total mean squared error
    def rmse(self):

        log_error = 0
        rmse_error = 0
        for i in range(0, self.N, 500):
            
            x_arr = self.data[i: i+500][:, :-1]
            rating = self.data[i: i+500][:, -1].astype(int)

            pdf = self.get_rating(x_arr)[1]
            
            pre_score = pdf.argmax(axis=1) + 1
            pre_score_dist = pdf[np.arange(len(rating)), rating-1]
            
            log_error += np.sum(np.log(pre_score_dist))

            rmse_error += np.sum(pow(rating - pre_score, 2))
        #print(log_error)
        log_error /= self.N
        #print(log_error)
        log_error -= self.reg_pu*np.sum(self.P**2) / 2 + self.reg_t*np.sum(self.t**2) / 2 + \
                     self.reg_beta*np.sum(self.beta**2) / 2
        
        return log_error, np.sqrt(rmse_error/self.N)
    
    # Stochastic gradient descent to get optimized P and Q matrix
    def sgd(self, x_input, trues, cdfs_s, prediction):

        batch = len(trues)
        bigger_r = 1 - cdfs_s[np.arange(batch), trues]
        smaller_r = cdfs_s[np.arange(batch), trues-1]
        diff = bigger_r - smaller_r
        
        # self.b_u += self.lr_bu * (-diff / batch - self.reg_bu*self.b_u)

        tmp_p = -np.dot(x_input.T, diff)
        self.P += self.lr_pu * (tmp_p / batch - self.reg_pu*self.P)

  #      t = self.t
       # print(np.sum(diff) / batch)
        
        #unique_r, update_t = npi.group_by(trues).mean(diff, axis=0)
       # print(update_t)
        #print('diff = {0:.4f}, t = {1:.4f}'.format(np.mean(diff), self.t[0]))
        self.t += self.lr_t * (np.mean(diff) - self.reg_t*self.t)
        beta1, beta2, beta3 = self.beta.copy()
        
        update_beta1 =  np.sum(-cdfs_s[trues==5, 4]) + np.sum(diff[trues==3]) + np.sum(diff[trues==4]) + \
                        np.sum(cdfs_s[trues==2, 2]*(1-cdfs_s[trues==2, 2])/(cdfs_s[trues==2, 2]-cdfs_s[trues==2, 1]))
                        
        update_beta1 *= np.exp(beta1)
        
        update_beta2 =  np.sum(cdfs_s[trues==3, 3]*(1-cdfs_s[trues==3, 3])/(cdfs_s[trues==3, 3]-cdfs_s[trues==3, 2])) + \
                        np.sum(diff[trues==4]) + np.sum(-cdfs_s[trues==5, 4])
        update_beta2 *= np.exp(beta2)
        
        update_beta3 =  np.sum(cdfs_s[trues==4, 4]*(1-cdfs_s[trues==4, 4])/(cdfs_s[trues==4, 4]-cdfs_s[trues==4, 3])) + \
                        np.sum(-cdfs_s[trues==5, 4])
        update_beta3 *= np.exp(beta3)

        update_beta = np.asarray([update_beta1, update_beta2, update_beta3])
        update_beta /= batch
        self.beta += self.lr_beta * (update_beta - self.reg_beta * self.beta)
    
        return self
    
    # Ratings for user i and moive j
    def get_rating(self, x_input):

        prediction = np.dot(x_input, self.P) #+ self.b_u   
        cdfs = 1 / (1+np.exp(prediction[:, np.newaxis] - self.thresholds_all()[np.newaxis, :]))
        score_dist = cdfs[:, 1:] - cdfs[:, :-1] 
        return cdfs, score_dist, prediction

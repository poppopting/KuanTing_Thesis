import os
import numpy as np
import pandas as pd
import pickle
from tqdm.auto import  tqdm

N = 50000
dim = 20
r = 5

try :
    os.mkdir('simulation_data')
except FileExistsError:
    pass

for i in tqdm(range(20)):
    
    X = np.random.normal(size=(N, dim))
    P = np.random.randn(dim)
    t = np.random.randn(1)
    beta = np.random.randn(r - 2)
    thresold = np.append(t, np.exp(beta)).cumsum()

    y = np.dot(X, P)

    # compute distribution functions
    F = 1 /(1 + np.exp(y[:, np.newaxis] - thresold)) 
    F = np.append(np.zeros(F.shape[0])[:, np.newaxis], np.append(F, np.ones(F.shape[0])[:, np.newaxis], axis=1), axis=1)
    
    mass = np.diff(F, axis=1)
    score = np.apply_along_axis(func1d=lambda x: np.asscalar(np.random.choice(list(range(r)), size=1, p=x)),
                                arr=mass, axis=1) + 1
    data = pd.DataFrame(X)
    data.loc[:, 'score'] = score
    
    params = {'P': P, 
              't': t,
              'beta': beta}
    
    pickle.dump(params, open('simulation_data\parameters_%d.pkl'%i, 'wb'))
    data.to_csv('simulation_data\data_%d.csv' % i, index=False)

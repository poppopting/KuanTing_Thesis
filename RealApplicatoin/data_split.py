import os
import sys
package_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
sys.path.append(package_path)
from Ting_MF import my_split
import pandas as pd
import numpy as np

print('Now begin Train/ Test split of MovieLens-100k')
data_100k = pd.read_csv(r'ml-100k\u.data', sep='\t', usecols=[0,1,2], names=['user', 'item', 'ratings'])
train_100k, test_100k = my_split.rec_train_test_split(data_100k, size=0.2)
train_100k.to_csv(r'ml-100k\train_100k.csv', index=False)
test_100k.to_csv(r'ml-100k\test_100k.csv', index=False)
print()
print('Now begin Train/ Test split of MovieLens-1M')
data_1m = pd.read_csv(r'ml-1m\ratings.dat',sep='::', engine= 'python', usecols=[0,1,2], names=['user', 'item', 'ratings'])
train_1m, test_1m = my_split.rec_train_test_split(data_1m, size=0.2)
train_1m.to_csv(r'ml-1m\train_1m.csv', index=False)
test_1m.to_csv(r'ml-1m\test_1m.csv', index=False)

properties = pd.DataFrame()
properties['Dataset'] = ['MovieLens-100k', 'MovieLens-1M']
properties['NumberOfUsers'] = [data_100k.user.nunique(), data_1m.user.nunique()]
properties['NumberOfItems'] = [data_100k.item.nunique(), data_1m.item.nunique()]
properties['|Train|'] = [len(train_100k), len(train_1m)]
properties['|Test|'] = [len(test_100k), len(test_1m)]
print(properties)
print('Save properties of dataset as DataProperties(table4).csv')
properties.to_csv('tables\DataProperties(table4).csv', index=False)

import os
import wget
import zipfile 

wget.download('http://files.grouplens.org/datasets/movielens/ml-100k.zip')
zf = zipfile.ZipFile('ml-100k.zip')
zf.extractall()
zf.close()
os.remove('ml-100k.zip')

wget.download('http://files.grouplens.org/datasets/movielens/ml-1m.zip')
zf = zipfile.ZipFile('ml-1m.zip')
zf.extractall()
zf.close()
os.remove('ml-1m.zip')

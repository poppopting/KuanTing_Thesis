import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data_100k = pd.read_csv(r'ml-100k\u.data', sep='\t', usecols=[0,1,2], names=['user', 'item', 'ratings'])
data_1m = pd.read_csv(r'ml-1m\ratings.dat',sep='::', engine= 'python', usecols=[0,1,2], names=['user', 'item', 'ratings'])

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
sns.countplot(data_100k.ratings)
plt.title('Ratings distribution of MovieLens-100k', fontsize=18)
plt.xticks(fontsize=12)
plt.xlabel('Ratings', fontsize=15)
plt.yticks(fontsize=12)
plt.ylabel('Count', fontsize=15)
plt.subplot(1,2,2)
sns.countplot(data_1m.ratings)
plt.title('Ratings distribution of MovieLens-1M', fontsize=18)
plt.xticks(fontsize=12)
plt.xlabel('Ratings', fontsize=15)
plt.yticks(fontsize=12)
plt.ylabel('Count', fontsize=15)
plt.savefig('ratings distribution.jpg')
plt.show()

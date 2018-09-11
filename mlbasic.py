import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns
sns.set(color_codes=True)
#%matplotlib inline%

df = pd.read_csv('iris.data',header=None)

col_name =['Sepal_length','Sepal_width','Petal_length','Petal_width','Species']
df.columns = col_name
print(df.head())
print(df.describe())
print(df.info())

iris = sns.load_dataset('iris')
#print(iris.head())
#print(iris.describe())
#print(iris.info())

print(iris.groupby('species').size())
###Visualization####

sns.pairplot(iris,hue='species',size=2,aspect=1)
##Histogram Method##
iris.hist(edgecolor='black',linewidth=1.4,figsize=(12,8))

plt.figure(figsize=(12,8))
plt.subplot(2,2,1)
sns.violinplot(x='species',y='sepal_length',data=iris)
plt.subplot(2,2,2)
sns.violinplot(x='species',y='sepal_width',data=iris)
plt.subplot(2,2,3)
sns.violinplot(x='species',y='petal_length',data=iris)
plt.subplot(2,2,4)
sns.violinplot(x='species',y='petal_width',data=iris)

iris.boxplot(by='species',figsize=(12,8));


pd.scatter_matrix(iris,figsize=(12,10));
plt.show()

#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn


# In[40]:


df = pd.read_csv('heart.csv')
df


# In[10]:


df.describe()


# In[13]:


nan_df = df.isna()
nan_df.sum()


# In[21]:


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
threshold = 1.5
lower_bound = Q1 - threshold * IQR
upper_bound = Q3 + threshold * IQR
outliers = (df < lower_bound) | (df > upper_bound)
df_outliers = df.loc[outliers.any(axis=1)]
df_outliers


# In[24]:


df = df[~outliers.any(axis=1)]
df


# In[31]:


corrMatrix = df.corr()
plt.figure(figsize=(12, 12))
sn.heatmap(corrMatrix, annot=True)
plt.show()


# In[33]:


feature_list = df.columns.tolist()
feature_list.remove("target")

cols = 5
n=len(feature_list)
rows = int(np.ceil(n/cols))


plt.figure(figsize=(cols*4 , rows*4))

i=0
for val in feature_list:
    i += 1
    plt.subplot(rows, cols, i)
    sn.kdeplot(x=val,data=df)
    plt.ylabel(" ")
    plt.xlabel(" ")
    plt.title(val +"_kdeplot")   


# In[53]:


fig, axs = plt.subplots(nrows=3, ncols=5, figsize=(20, 10))

# loop over each column in the dataframe and plot a histogram on the corresponding subplot
for i, column in enumerate(df.columns):
    row = i // 5
    col = i % 5
    axs[row, col].hist(df[column])
    axs[row, col].set_title(column)


# In[48]:


range(1,13)


# In[ ]:





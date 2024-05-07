#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Data Analytics I Create a Linear Regression Model using Python/R to predict home prices using Boston Housing Dataset 
# (https://www.kaggle.com/c/boston-housing). The Boston Housing dataset contains information about various houses in Boston
# through different parameters. There are 506 samples and 14 feature variables in this dataset. 
# The objective is to predict the value of prices of the house using the given features. 


# In[14]:


import pandas as pd


# In[15]:


import numpy as np


# In[16]:


import matplotlib.pyplot as plt


# In[17]:


from sklearn.model_selection import train_test_split


# In[18]:


from sklearn.linear_model import LinearRegression


# In[19]:


from sklearn.metrics import mean_squared_error


# In[20]:


df=pd.read_csv("Boston.csv")


# In[21]:


df


# In[24]:


df.columns


# In[32]:


x=df[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis',
       'rad', 'tax', 'ptratio', 'black', 'lstat']]
y=df['medv']


# In[34]:


x


# In[35]:


y


# In[41]:


x_train, x_test,y_train, y_test=train_test_split(x,y,test_size=0.25,random_state=42)


# In[43]:


model=LinearRegression()
model.fit(x_train,y_train)


# In[45]:


y_predict=model.predict(x_test)


# In[46]:


y_predict


# In[47]:


model.score(x_train, y_train)


# In[48]:


model.score(x_test, y_test)


# In[50]:


np.sqrt(mean_squared_error(y_test, y_predict))


# In[ ]:





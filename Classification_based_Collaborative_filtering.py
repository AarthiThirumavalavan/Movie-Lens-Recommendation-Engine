
# coding: utf-8

# In[40]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from sklearn.linear_model import LogisticRegression


# In[39]:


bank_full = pd.read_csv("C:/Users/thirumav/Desktop/Python_exercises/Recommendation system_lynda/Ex_Files_Intro_Python_Rec_Systems/Ex_Files_Intro_Python_Rec_Systems/Exercise Files/02_01/bank_full_w_dummy_vars.csv")


# In[3]:


bank_full.head()


# In[4]:


bank_full.info()


# In[47]:


X = bank_full.iloc[:,18:37]


# In[49]:


y = bank_full.iloc[:,17]


# In[50]:


LogReg = LogisticRegression()


# In[51]:


LogReg.fit(X,y)


# In[52]:


new_user = [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]]


# In[53]:


y_pred = LogReg.predict(new_user)
y_pred


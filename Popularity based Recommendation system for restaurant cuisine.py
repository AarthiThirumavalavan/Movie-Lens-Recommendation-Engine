
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


frame = pd.read_csv("C:/Users/thirumav/Desktop/Python_exercises/Recommendation system_lynda/Ex_Files_Intro_Python_Rec_Systems/Ex_Files_Intro_Python_Rec_Systems/Exercise Files/01_02/rating_final.csv")


# In[3]:


cuisine = pd.read_csv("C:/Users/thirumav/Desktop/Python_exercises/Recommendation system_lynda/Ex_Files_Intro_Python_Rec_Systems/Ex_Files_Intro_Python_Rec_Systems/Exercise Files/01_02/chefmozcuisine.csv") 


# In[4]:


frame.head()


# In[5]:


cuisine.head()


# In[ ]:


#Recommendation based on counting


# In[9]:


rating_count = pd.DataFrame(frame.groupby("placeID")["rating"].count())
rating_count.sort_values("rating", ascending = False).head()


# In[10]:


most_rated_places = pd.DataFrame([135085,132825,135032,135052,132834], index = np.arange(5), columns = ['placeID'])


# In[14]:


summary = pd.merge(most_rated_places, cuisine, on="placeID")
summary


# In[16]:


cuisine['Rcuisine'].describe()


# In[ ]:


#Conclusion : Top value is Mexican and in summary, 2 Rcuisine values are Mexican. This means most popular cuisine is Mexican in the dataset.


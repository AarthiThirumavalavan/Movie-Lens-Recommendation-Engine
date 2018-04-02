
# coding: utf-8

# In[1]:


#ITEM BASED RECOMMENDATION SYSTEM USING PEARSON'S CORRELATION


# In[2]:


import pandas as pd
import numpy as np


# In[3]:


frame = pd.read_csv("C:/Users/thirumav/Desktop/Python_exercises/Recommendation system_lynda/Ex_Files_Intro_Python_Rec_Systems/Ex_Files_Intro_Python_Rec_Systems/Exercise Files/01_03/rating_final.csv")


# In[5]:


cuisine = pd.read_csv("C:/Users/thirumav/Desktop/Python_exercises/Recommendation system_lynda/Ex_Files_Intro_Python_Rec_Systems/Ex_Files_Intro_Python_Rec_Systems/Exercise Files/01_03/chefmozcuisine.csv")


# In[8]:


geoinfo = pd.read_csv("C:/Users/thirumav/Desktop/Python_exercises/Recommendation system_lynda/Ex_Files_Intro_Python_Rec_Systems/Ex_Files_Intro_Python_Rec_Systems/Exercise Files/01_03/geoplaces2.csv", encoding = 'latin-1')


# In[9]:


frame.head()


# In[10]:


geoinfo.head()


# In[12]:


places = geoinfo[["placeID","name"]]
places.head()


# In[13]:


cuisine.head()


# In[14]:


#Grouping and Rating


# In[17]:


rating = pd.DataFrame(frame.groupby("placeID")["rating"].mean())
rating.head()


# In[20]:


rating["rating_count"] = pd.DataFrame(frame.groupby("placeID")["rating"].count())
rating.head()


# In[21]:


rating.describe()


# In[23]:


rating = rating.sort_values("rating_count", ascending = False)
rating.head()


# In[27]:


places[places["placeID"]==135085]


# In[28]:


cuisine.head()


# In[29]:


cuisine[cuisine["placeID"]==135085]


# In[30]:


#Preparing Data for Analysis


# In[33]:


#Finding the user rating for each of the placeIDs in frame dataframe


# In[31]:


places_crosstab = pd.pivot_table(data = frame, values = 'rating', index = "userID", columns = 'placeID')


# In[32]:


places_crosstab.head()


# In[41]:


Tortas_ratings = places_crosstab[135085]
Tortas_ratings[Tortas_ratings >= 0]


# In[43]:


#Evaluating Similarity Based on Correlation


# In[44]:


similar_to_Tortas = places_crosstab.corrwith(Tortas_ratings)
corr_Tortas = pd.DataFrame(similar_to_Tortas, columns = ["PearsonR"])
corr_Tortas.dropna(inplace = True)
corr_Tortas.head()


# In[45]:


Tortas_Corr_summary = corr_Tortas.join(rating['rating_count'])


# In[46]:


Tortas_Corr_summary[Tortas_Corr_summary["rating_count"]>=10].sort_values("PearsonR", ascending = False).head(10)


# In[47]:


places_corr_Tortas = pd.DataFrame([135085, 132754, 135045, 135062, 135028, 135042, 135046], index = np.arange(7), columns =["placeID"] )


# In[48]:


summary = pd.merge(places_corr_Tortas, cuisine, on= "placeID")


# In[49]:


summary


# In[50]:


places[places["placeID"] ==135046]


# In[52]:


cuisine['Rcuisine'].describe()


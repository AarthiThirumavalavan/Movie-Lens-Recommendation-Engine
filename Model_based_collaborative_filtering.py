
# coding: utf-8

# In[1]:


#MODEL BASED COLLABORATIVE FILTERING SYSTEMS


# In[2]:


#SVD MATRIX FACTORIZATION


# In[3]:


import pandas as pd
import numpy as np
import sklearn
from sklearn.decomposition import TruncatedSVD


# In[4]:


columns = ["user_id", "item_id", "rating", "timestamp"]


# In[6]:


frame = pd.read_csv("C:/Users/thirumav/Desktop/Python_exercises/MovieLens 1M/ml-100k/ml-100k/u.data", sep='\t', names=columns)


# In[7]:


frame.head()


# In[8]:


columns = ["item_id", "movie title", "release date", "video release date", "IMDB URL", "unknown", "Action", "Adventure", "Animation",
          "Childrens'", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
           "Sci-Fi", "Thriller", "War", "Western"]


# In[9]:


movies = pd.read_csv("C:/Users/thirumav/Desktop/Python_exercises/MovieLens 1M/ml-100k/ml-100k/u.item", sep= '|', names = columns, encoding = 'latin-1')


# In[15]:


movie_names = movies[["item_id", "movie title"]]
movie_names.head()


# In[17]:


combined_movies_data = pd.merge(frame, movie_names, on = "item_id")
combined_movies_data.head()


# In[20]:


combined_movies_data.groupby("item_id")["rating"].count().sort_values(ascending=False).head()


# In[22]:


Filter = combined_movies_Data["item_id"]==50
combined_movies_data[Filter]["movie title"].unique()


# In[23]:


#Building a Utility Matrix


# In[28]:


rating_crosstab = combined_movies_data.pivot_table(values = "rating", index = "user_id", columns = "movie title", fill_value = 0)
rating_crosstab.head()


# In[29]:


#Transposing the Matrix


# In[32]:


rating_crosstab.shape


# In[33]:


X = rating_crosstab.values.T


# In[34]:


X.shape


# In[35]:


#Decomposing the Matrix


# In[36]:


SVD = TruncatedSVD(n_components = 12, random_state = 17)


# In[39]:


resultant_matrix = SVD.fit_transform(X)
resultant_matrix.shape


# In[40]:


#Generating Correlation Matrix   


# In[41]:


corr_mat = np.corrcoef(resultant_matrix)


# In[42]:


corr_mat.shape


# In[43]:


#Isolating Starwars from correlation matrix


# In[46]:


movies_names = rating_crosstab.columns
movies_list = list(movies_names)


# In[47]:


star_wars = movies_list.index("Star Wars (1977)")


# In[48]:


star_wars


# In[51]:


corr_star_wars = corr_mat[star_wars]
corr_star_wars.shape #represents correlation of each movie with StarWars


# In[52]:


#Recommending a Highly Correlated Movie


# In[54]:


list(movies_names[(corr_star_wars < 1.0) & (corr_star_wars > 0.9)])


# In[55]:


list(movies_names[(corr_star_wars < 1.0) & (corr_star_wars > 0.95)])


# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 11:58:40 2018

@author: thirumav
"""

import graphlab
import pandas
#Reading Rating file
user_data = pd.read_table("C:/Users/thirumav/Desktop/Python_exercises/MovieLens 1M/ml-100k/ml-100k/u.data",
                           header = None, names =["UserID", "ItemID", "Rating", "Timestamp"])
user_data = user_data.set_index("UserID")

#Reading Item file
user_item = pd.read_table("C:/Users/thirumav/Desktop/Python_exercises/MovieLens 1M/ml-100k/ml-100k/u.item",
                           header = None, names =["Movie ID", "Movie Title", "Release date", "Video Release date",
                                                  "IMDb URL", "Unknown", "Action", "Adventure", "Animation",
                                                  "Children's", "Comedy", "Crime", "Documentary", "Drama",
                                                  "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery", 
                                                  "Romance", "Sci-Fi", "Thriller", "War", "Western"], sep="|", encoding = "latin-1")
user_item = user_item.set_index("Movie ID")

#Reading the User file
user_user = pd.read_table("C:/Users/thirumav/Desktop/Python_exercises/MovieLens 1M/ml-100k/ml-100k/u.user",
                           header = None, names =["UserID", "Age", "Gender", "Occupation", "Zip code"], sep="|")
user_user = user_user.set_index("UserID")

#Reading the train and test set for Ratings file
ratings_train = pd.read_table("C:/Users/thirumav/Desktop/Python_exercises/MovieLens 1M/ml-100k/ml-100k/ua.base",
                           header = None, names =["UserID", "ItemID", "Rating", "Timestamp"])
ratings_train = ratings_train.set_index("UserID")

ratings_test = pd.read_table("C:/Users/thirumav/Desktop/Python_exercises/MovieLens 1M/ml-100k/ml-100k/ua.test",
                           header = None, names =["UserID", "ItemID", "Rating", "Timestamp"])
ratings_test = ratings_test.set_index("UserID")

#Using GraphLab
train_data = graphlab.SFrame(ratings_train)



#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import os
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# In[2]:


books = pd.read_csv("books.csv")


# In[3]:


books = books[(books[['average_rating','ratings_count','text_reviews_count']] != '0').all(axis=1)]
books.drop(books.index[books['authors'] == 'NOT A BOOK'], inplace = True)


# In[4]:


content_data = books[['title','authors','average_rating']]
content_data = content_data.astype(str)


# In[5]:


content_data['content'] = content_data['title'] + ' ' + content_data['authors'] + ' ' + content_data['average_rating']


# In[6]:


content_data = content_data.reset_index()
indices = pd.Series(content_data.index, index=content_data['title'])


# In[7]:


select_books = books['title'].unique()


# In[8]:


count = CountVectorizer(stop_words='english')
count_matrix = count.fit_transform(content_data['content'])

cosine_sim_content = cosine_similarity(count_matrix, count_matrix)


# In[9]:


def get_recommendations(title, cosine_sim=cosine_sim_content):
    idx = indices[title]

    # Get the pairwsie similarity scores of all books with that book
    sim_scores = list(enumerate(cosine_sim_content[idx]))

    # Sort the books based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar books
    sim_scores = sim_scores[1:11]

    # Get the book indices
    book_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar books
    return list(content_data['title'].iloc[book_indices])


# In[10]:


def book_shows(book):
    for book in book:
        print(book)


# In[11]:


books4 = get_recommendations(input("Enter the book"), cosine_sim_content)
book_shows(books4)


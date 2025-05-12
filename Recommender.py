#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import Libraries
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


#to ignore warnings
import warnings
warnings.filterwarnings('ignore')


# In[3]:


#import data
books=pd.read_csv('Books.csv')
users=pd.read_csv('Users.csv')
ratings=pd.read_csv('Ratings.csv')


# In[4]:


books.head()


# In[5]:


users.head()


# In[6]:


ratings.head()


# In[7]:


books.shape


# In[8]:


ratings.shape


# In[9]:


users.shape


# In[10]:


#Looking for null values in books data
books.isnull().sum()


# In[11]:


#Drop those nulls
books=books.dropna()


# In[12]:


books.isnull()


# In[13]:


books.isnull().sum()


# In[14]:


users.isnull().sum()


# In[15]:


users=users.dropna()


# In[16]:


users.isnull().sum()


# In[17]:


ratings.isnull().sum()


# In[18]:


books.shape


# In[19]:


users.shape


# In[20]:


ratings.shape


# In[21]:


#checking for duplicates
books.duplicated().sum()


# In[22]:


users.duplicated().sum()


# In[23]:


ratings.duplicated().sum()


# In[24]:


#Unique count
books.nunique()


# In[25]:


users.nunique()


# In[26]:


ratings.nunique()


# In[27]:


users.head()


# In[28]:


ratings.head()


# In[29]:


np.sort(ratings['Book-Rating'].unique())


# In[30]:


books.info()


# In[31]:


books.columns


# In[32]:


#convert year of publication into int
books['Year-Of-Publication']=books['Year-Of-Publication'].astype('int32')


# In[33]:


books.info()


# In[34]:


ratings.info()


# In[35]:


users.info()


# ##Popularity Based Recommendation System

# In[36]:


books.head()


# In[37]:


ratings.head()


# In[38]:


#Joining books and user ratings into a table
books_with_ratings=ratings.merge(books, on = 'ISBN')


# In[39]:


books_with_ratings.head()


# In[40]:


books_with_ratings.shape


# In[41]:


popular_df=books_with_ratings.groupby('Book-Title').agg(num_rating=('Book-Rating','count'),avg_rating=('Book-Rating', 'mean'))


# In[42]:


popular_df=popular_df.reset_index()


# In[43]:


popular_df


# In[44]:


popular_df.sort_values('num_rating',ascending=False)


# In[45]:


#Popularity is based on the number of people reasd the book ('num_rating' > 300)
#It is based on the rating it got.
popular_df=popular_df[popular_df['num_rating']>300].sort_values('avg_rating',ascending=False)


# In[46]:


popular_df


# In[47]:


popular_df=popular_df.head(50)


# In[48]:


popular_df


# In[49]:


popular_df=popular_df.merge(books, on = 'Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','num_rating','avg_rating']]


# In[50]:


popular_df


# ###Colaborative Filtering
# -Similar book prediction based on users feedback

# In[51]:


books_with_ratings.head()


# In[52]:


#Grouping based on user-id tells the no.of books rated by each user
x=books_with_ratings.groupby('User-ID').count()
x


# In[53]:


x.index


# In[54]:


x.shape


# In[55]:


# Select only users who atleast gave feddback for 200 books (Power Users)
x=x['Book-Rating'] > 200
x


# In[56]:


power_users=x[x].index


# In[57]:


power_users


# In[58]:


power_users.shape


# In[59]:


power_users.sort_values()


# In[60]:


# sELECTING ONLY RECORDS OF POWER_USERS
filtered_ratings=books_with_ratings[books_with_ratings['User-ID'].isin(power_users)]


# In[61]:


filtered_ratings


# In[62]:


# so now considering the best users (atleast 200 books feedback) grouping them on the book title
y=filtered_ratings.groupby('Book-Title').count()


# In[63]:


y


# In[64]:


# The above y data frame tells hw many users have read that book.
y.sort_values('User-ID',ascending=False)


# In[65]:


y=y['User-ID'] >=50
y


# In[66]:


famous_books=y[y].index


# In[67]:


famous_books


# In[68]:


final_ratings=filtered_ratings[filtered_ratings['Book-Title'].isin(famous_books)]


# In[69]:


final_ratings


# In[70]:


#pivot table giving the rating for each book from the user
#Book row with uer-ID as Column
pt=final_ratings.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating')


# In[71]:


pt


# In[72]:


pt=pt.fillna(0)


# In[73]:


pt


# In[74]:


similarity_scores=cosine_similarity(pt)
similarity_scores


# In[75]:


type(similarity_scores)


# In[76]:


df_temp=pd.DataFrame(similarity_scores)


# In[77]:


df_temp


# In[78]:


pt.index


# In[79]:


df_temp_name=df_temp


# In[80]:


df_temp_name.index=pt.index
df_temp_name.columns=pt.index


# In[81]:


df_temp_name


# In[82]:


def recommend(book_name):
    index=np.where(pt.index == book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x : x[1],reverse=True)[1:6]
    #Lets create List and in that lies i want at populate with the book information
    #book author book-title image url
    #Empty list
    data=[]
    for i in similar_items:
        item=[]
        temp_df=books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        data.append(item)
    return data


# In[83]:


recommend('1984')


# In[84]:


recommend('Animal Farm')


# In[85]:


#Export data and model to pkl


# In[86]:


#Import Pickle and dum[p the data models
import pickle as pkl
pkl.dump(popular_df,open('popular.pkl','wb')) # Popularity based recommender system


# In[87]:


pkl.dump(books,open('books.pkl','wb')) # Going to give book data
pkl.dump(pt,open('pt.pkl','wb')) #books and users feedback
pkl.dump(similarity_scores, open('similarity_scores.pkl','wb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





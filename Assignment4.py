#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.vector_ar.var_model import VAR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import scipy.stats as stats


# In[4]:


# Importing News Headline Dataset
news_data = pd.read_csv('C:/Users/SHIKHA/Documents/MBA/Skill_Building/Data Science/Spark Foundation/News.csv')
news_data.head()


# In[5]:


# Pre-processing of News Dataset
news_data['published_date'] = np.array([str(str(str(x)[:4]) + '/' + str(str(x)[4:6]) + '/' + str(str(x)[6:])) for x in news_data['publish_date']])
news_data.head()


# In[6]:


news_data = news_data.drop('publish_date', axis=1)
news_data['published_date'] = pd.to_datetime(news_data['published_date'])
news_data = news_data[['published_date', 'headline_text']]
news_data.columns = ['published_date', 'headline']
news_data.head()


# In[9]:


# Aggregating the news headlines having same dates
dict_news = {}
temp = news_data.loc[0, 'published_date']
temp2 = str(news_data.loc[0, 'headline'])
for x in range(1, len(news_data)):
    if news_data.loc[x, 'published_date']==temp:
        temp2 += '. ' + str(news_data.loc[x, 'headline'])
    else:
        dict_news[news_data.loc[x-1, 'published_date']] = temp2
        temp2 = ""
        temp = news_data.loc[x, 'published_date']


# In[19]:


indexes = np.arange(0, len(dict_news))
df_news = pd.DataFrame(indexes)
df_news.head()


# In[20]:


df_news['Published_Date'] = dict_news.keys()
df_news.head()


# In[21]:


l = []
for i in dict_news.keys():
    l.append(dict_news[i])
df_news['Headline'] = np.array(l)
df_news.head()


# In[22]:


df_news = df_news.drop(0, axis=1)


# In[24]:


# Performing Sentiment Analysis
polarity = []
subjectivity = []
tuples = []
for i in df_news['Headline'].values:
    my_valence = TextBlob(i)
    tuples.append(my_valence.sentiment)


# In[25]:


for i in tuples:
    polarity.append(i[0])
    subjectivity.append(i[1])


# In[26]:


df_news['Polarity'] = np.array(polarity)
df_news['Subjectivity'] = np.array(subjectivity)


# In[27]:


temp = ['Positive', 'Negative', 'Neutral']
temp1 = ['Factual', 'Public']
polarity = []
subjectivity = []
for i in range(len(df_news)):
    pol = df_news.iloc[i]['Polarity']
    sub = df_news.iloc[i]['Subjectivity']
    if pol>=0:
        if pol>=0.2:
            polarity.append(temp[0])
        else:
            polarity.append(temp[2])
    else:
        if pol<=-0.2:
            polarity.append(temp[1])
        else:
            polarity.append(temp[2])
    
    if sub>=0.4:
        subjectivity.append(temp1[1])
    else:
        subjectivity.append(temp1[0])


# In[28]:


df_news['Sentiment'] = polarity
df_news['Opinion'] = subjectivity
df_news.head()


# In[29]:


# Sentiments distribution
plt.figure(figsize=(6,4))
df_news['Subjectivity'].hist()
plt.show()
plt.figure(figsize=(6,4))
df_news['Polarity'].hist()
plt.show()


# In[ ]:





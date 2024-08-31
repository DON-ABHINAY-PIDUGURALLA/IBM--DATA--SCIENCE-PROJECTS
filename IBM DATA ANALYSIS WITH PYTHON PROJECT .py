#!/usr/bin/env python
# coding: utf-8

# In[65]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')


# ## NOW I AM IMPORTING DATA SETS 

# In[66]:


file_name='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DA0101EN/coursera/project/kc_house_data_NaN.csv'
df=pd.read_csv(file_name)


# In[51]:


df.head()
df


# # QUESTION 1 ( MODULE 1)

# In[52]:


df.dtypes


# In[53]:


df.describe()


# # QUESTION 2 ( MODULE 2)

# In[54]:


df.drop(['id', 'Unnamed: 0'], axis=1, inplace=True)
df.describe()


# In[55]:


print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


# In[56]:


mean=df['bedrooms'].mean()
df['bedrooms'].replace(np.nan,mean, inplace=True)


# In[57]:


mean=df['bathrooms'].mean()
df['bathrooms'].replace(np.nan,mean, inplace=True)


# In[58]:


print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())


# # QUESTION 3 (MODULE 3)

# In[59]:


y = df['floors'].value_counts().to_frame()
y


# # QUESTION 4

# In[60]:


sns.boxplot(x = 'waterfront',  y = 'price', data = df)


# # QUESTION 5 

# In[61]:


sns.regplot(x = 'sqft_above', y = 'price', data = df)


# In[62]:


numeric_df = df.select_dtypes(include=['number'])


correlation_with_price = numeric_df.corr()['price'].sort_values()
print(correlation_with_price)


# #### MODULE 4

# In[63]:


X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
lm.score(X, Y)


# # QUESTION 6 

# In[64]:


lm.fit(df[['sqft_living']],df['price'])
yhat_a = lm.predict(df[['sqft_living']])
print(yhat_a)
lm.score(df[['sqft_living']],df['price'])


# # QUESTION 7

# In[ ]:


features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms",
           "sqft_living15","sqft_above","grade","sqft_living"]     
yhat = lm.predict(df[features])
yhat


# In[ ]:


array([283850.64176653, 662015.89176653, 307084.89176653, ...,
       303822.64176653, 428176.14176653, 303694.64176653])


# In[ ]:


lm = LinearRegression()
lm.fit(df[features], df['price'])
lm.score(df[features], df['price'])


# In[ ]:


0.657679183672129


# # QUESTION 8

# In[72]:


Input=[('scale',StandardScaler()),('polynomial', PolynomialFeatures(include_bias=False)),('model',LinearRegression())]


# In[74]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

# Define your features and target
W = df[features]
y = df['price']

# Create a pipeline with an imputer step
Input = [
    ('imputer', SimpleImputer(strategy='mean')),  # Replace missing values with the mean of the column
    ('scale', StandardScaler()),                 # Standardize features
    ('polynomial', PolynomialFeatures(include_bias=False)),  # Add polynomial features
    ('model', LinearRegression())                # Linear regression model
]

pipe = Pipeline(Input)

# Fit the model
pipe.fit(W, y)

# Print the model score
print(pipe.score(W, y))


# #### MODULE 5

# In[76]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
print("done")


# In[77]:


features =["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]    
X = df[features]
Y = df['price']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=1)


print("number of test samples:", x_test.shape[0])
print("number of training samples:",x_train.shape[0])


# # QUESTION 9 

# In[81]:


from sklearn.linear_model import Ridge


# In[83]:


from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge

# Impute missing values in x_test
imputer = SimpleImputer(strategy='mean')  # You can choose 'mean', 'median', 'most_frequent', or a constant.
x_test_imputed = imputer.fit_transform(x_test)

# Fit Ridge regression model
Ridge_test = Ridge(alpha=0.1)
Ridge_test.fit(x_test_imputed, y_test)

# Score the model
score = Ridge_test.score(x_test_imputed, y_test)
print(f"Ridge Regression Score: {score}")


# # QUESTION 10 

# In[85]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

# Impute missing values
imputer = SimpleImputer(strategy='mean')  # You can choose 'mean', 'median', or 'most_frequent'
x_train_imputed = imputer.fit_transform(x_train)
x_test_imputed = imputer.transform(x_test)

# Apply Polynomial Features
pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train_imputed)
x_test_pr = pr.transform(x_test_imputed)

# Fit Ridge regression model
Ridge_test = Ridge(alpha=0.1)
Ridge_test.fit(x_train_pr, y_train)
score = Ridge_test.score(x_train_pr, y_train)

print(f"Ridge Regression Score: {score}")


# In[ ]:





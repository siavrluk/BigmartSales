
# coding: utf-8

# The data scientists at BigMart have collected 2013 sales data for 1559 products across 10 stores in different cities. Also, certain attributes of each product and store have been defined. The aim is to build a predictive model and find out the sales of each product at a particular store.

# In[107]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# In[2]:


train = pd.read_csv('../Bigmart Sales/train.csv')
test = pd.read_csv('../Bigmart Sales/test.csv')


# # Data Exploration

# In[3]:


train.shape
test.shape


# In[4]:


train.columns


# In[5]:


test.columns


# From the data set website (https://datahack.analyticsvidhya.com/contest/practice-problem-big-mart-sales-iii/) we have the following description of the features:
# 
# | Variable | Description |
# | --- | --- |
# | Item_Identifier | Unique product ID | 
# | Item_Weight | Weight of product |
# | Item_Fat_Content | Whether the product is low fat or not |
# | Item_Visibility | The % of total display area of all products in a store allocated to the particular product |
# | Item_Type | The category to which the product belongs |
# | Item_MRP | Maximum Retail Price (list price) of the product |
# | Outlet_Identifier | Unique store ID |
# | Outlet_Establishment_Year | The year in which store was established |
# | Outlet_Size | The size of the store in terms of ground area covered |
# | Outlet_Location_Type | The type of city in which the store is located |
# | Outlet_Type | Whether the outlet is just a grocery store or some sort of supermarket |
# | Item_Outlet_Sales | Sales of the product in the particulat store. This is the outcome variable to be predicted |
# 

# In[6]:


train.head()


# Check for missing values:

# In[7]:


train.isnull().sum(axis=0)/len(train)*100


# In[8]:


test.isnull().sum(axis=0)/len(test)*100


# The only features with missing values in both train and test sets are 'Item_Weight' and 'Outlet_Size'. The percentage missing values are very similar for both set, with 17.17% missing 'Item_Weight' and 28.27% missing 'Outlet_Size'.

# Look at the distributions of train and test set:

# In[9]:


train.describe()


# In[10]:


test.describe()


# Note that in both sets the minimum of 'Item_Visibility' is 0 which seems to denote missing information. 

# Both sets seem to have the same distributions of the quantitative features so we will combine them in order to do the missing values imputation.

# In[11]:


train['Src'] = 'train'
test['Src'] = 'test'
data = pd.concat([train, test], ignore_index=True, sort = False)


# In[12]:


print(train.shape, test.shape, data.shape)


# In[13]:


data.apply(lambda x: len(x.unique()))


# As mentioned in the description of the problem, there are 1559 products being sold in 10 stores. 
# 

# Explore the frequency of all categorical features:

# In[14]:


cat_pred = [x for x in data.dtypes.index if data.dtypes[x] == 'object']
cat_pred = [x for x in cat_pred if x not in ['Item_Identifier', 'Outlet_Identifier','Src']]
for col in cat_pred:
    print('Frequency of the categories in', col)
    print(data[col].value_counts())
    print('')


# We see that in the feature 'Item_Fat_Content' we have the category 'Low fat' being coded as both 'Low Fat', 'LF', and 'low fat'. The same happens with the category 'Regular' being coded as both 'Regular' and 'reg'.

# Note that some of the categories in 'Item_Type' are non-edible but they still have a specified fat-content. We will try to extract information from 'Item_Identifier' and 'Item_Type'.

# Few of the categories in 'Item_Type' have been sold a very little number of times, e.g. 'Seafood', 'Breakfast', Starchy Foods', 'Others', and 'Breads' account for 0.63%, 1.31%, 1.89%, 1.97%, 2.55%, and 2.93%, resp. It might be a good idea to group them together.

# We also note that most of the stores are of type 'Supermarket Type1', followed by 'Grocery Store'. The stores of type 'Supermarket Type2' and 'Supermarket Type3' are relatively few compared to 'Supermarket Type1' so we might want to group them together.

# # Data Cleaning and Feature Engineering

# As already mentioned, the feature 'Item_Fat_Content' is no coded properly so we will correct this:

# In[15]:


data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF': 'Low Fat', 
                                                             'low fat': 'Low Fat', 
                                                             'reg': 'Regular'})


# Now we will address the missing values of 'Item_Weight' and 'Outlet_Size'. We will impute the missing values in 'Item_Weight' with the average weight of the items in the corresponding 'Item_Type' category.

# In[16]:


data['Item_Weight'] = data.groupby('Item_Type')['Item_Weight'].transform(lambda x: x.fillna(x.mean()))


# For 'Outlet_Size' we first check the type and the location of the stores:

# In[17]:


pd.crosstab( data['Outlet_Type'], data['Outlet_Size'])


# In[18]:


pd.crosstab( data['Outlet_Type'], data['Outlet_Location_Type'])


# We see that all stores of type 'Supermarket Type2' and 'Supermarket Type3' are of medium size and are location is of type 'Tier 3'. Therefore, we will group them together:

# In[19]:


data['Outlet_Type'] = data['Outlet_Type'].replace({'Supermarket Type2': 'Supermarket Type2/3',
                                                  'Supermarket Type3': 'Supermarket Type2/3'})


# Now we will impute the missing values in 'Outlet_Size' by the mode of 'Outlet_Size' for the corresponding 'Outlet_Type':

# In[20]:


pd.crosstab( data['Outlet_Type'], data['Outlet_Size'])


# In[21]:


mode = (data.groupby('Outlet_Type', as_index=False)
        .agg({'Outlet_Size': lambda x: x.mode()[0]}))
mode


# In[22]:


data.loc[(data['Outlet_Type'] == 'Grocery Store') & (data['Outlet_Size'].isnull()), 'Outlet_Size'] = 'Small'
data.loc[(data['Outlet_Type'] == 'Supermarket Type1') & (data['Outlet_Size'].isnull()), 'Outlet_Size'] = 'Small'
data.loc[(data['Outlet_Type'] == 'Supermarket Type2/3') & (data['Outlet_Size'].isnull()), 'Outlet_Size'] = 'Medium'


# As already mentioned, 0's in 'Item_Visibility' seem to denote missing values. We will impute this with the average 'Item_Visibility' of the items in the corresponding 'Item_Type' category.

# In[23]:


data['Item_Visibility'].replace(0, np.nan, inplace=True)


# In[24]:


data['Item_Visibility'] = data.groupby('Item_Type')['Item_Visibility'].transform(lambda x: x.fillna(x.mean()))


# Now we deal with the items that are non-edible but still have fat content. As mentioned, we will try to extract information from the 'Item_Identifier' and 'Item_Type'. We note that although each item has a unique 'Item_identifier', 

# In[25]:


data.groupby('Item_Type')['Item_Identifier'].unique()


# All 'Item_Identifier' codes start with 'FD', 'DR' or 'NC' (stand for Food, Drink, Non-consumable?). We will replace the 'Item_Fat_Content' for the non-consumable items by 'DNA' (does not apply). 

# In[26]:


data['Item_Identifier_new'] = data['Item_Identifier'].apply(lambda x: x[0:2])


# In[27]:


data.groupby('Item_Type')['Item_Identifier_new'].unique()


# In addition, items of type 'Fruits and Vegetables', 'Hard Drinks', 'Meat', 'Seafood' will also get 'DNA' for 'Item_Fat_Content'.

# In[28]:


data.loc[data['Item_Identifier_new'] == 'NC', 'Item_Fat_Content'] = 'DNA'
data.loc[data.Item_Type.isin(['Fruits and Vegetables', 'Hard Drinks', 'Meat', 'Seafood']), 'Item_Fat_Content'] = 'DNA'


# From 'Outlet_Establishment_Year' we will define a new variable 'Years_Operation' which is the number of years the respective store had been operating:

# In[29]:


data['Years_Operation'] = 2013 - data['Outlet_Establishment_Year']


# # One-Hot-Encoding

# For building the prediction model we will use the scikit.learn library. Since it only accepts numerical values, first we need to convert all categorical variables into dummy variables. Note that since we need 'Outlet_Identifier' for the submission, we first need to define a new variable for each store and then dummify it.

# In[30]:


LE = LabelEncoder()
data['Store'] = LE.fit_transform(data['Outlet_Identifier'])
cat_pred_dummy = ['Item_Fat_Content', 'Item_Type', 'Outlet_Location_Type', 'Outlet_Size', 'Outlet_Type']
data_dummy = data.copy()


for var in cat_pred_dummy:
    data_dummy[var] = LE.fit_transform(data_dummy[var])
    
data_dummy = pd.get_dummies(data_dummy, columns = ['Item_Fat_Content', 'Item_Type', 'Outlet_Location_Type', 
                                        'Outlet_Size', 'Outlet_Type', 'Store'])


# # Predictive models

# The first step is to split the data set back into train and test sets:

# In[31]:


train = data.loc[data['Src'] == 'train']
test = data.loc[data['Src'] == 'test']

train_dummy = data_dummy.loc[data_dummy['Src'] == 'train']
test_dummy = data_dummy.loc[data_dummy['Src'] == 'test']


# ## Benchmark

# We create the benchmark model by predicting the sales as the average sales of product of the same 'Item_Type'.

# In[32]:


mean_per_item_type = train.groupby('Item_Type')['Item_Outlet_Sales'].mean()

subm = test[['Item_Identifier', 'Outlet_Identifier']]
subm['Item_Outlet_Sales'] = np.nan
subm['Item_Outlet_Sales'] = data.groupby('Item_Type')['Item_Outlet_Sales'].transform(lambda x: x.fillna(x.mean()))

subm.to_csv("benchmark.csv",index=False)


# Public Leaderboard Score: 1775.33583643103

# In[33]:


test.head()


# In[34]:


subm.head()


# ## Linear Regression

# In[35]:


pred = [x for x in train_dummy.columns if x not in ['Item_Outlet_Sales','Item_Identifier', 'Outlet_Identifier', 
                                                     'Outlet_Establishment_Year', 'Src', 'Item_Identifier_new']]


# In[36]:


lin_regr = LinearRegression(normalize=True)
lin_regr.fit(train_dummy[pred], train_dummy['Item_Outlet_Sales'])
test_dummy['Item_Outlet_Sales'] = lin_regr.predict(test_dummy[pred])


# In[37]:


cv_score = np.sqrt(np.abs(cross_val_score(lin_regr, train_dummy[pred], train_dummy['Item_Outlet_Sales'], 
                cv = 3, scoring='neg_mean_squared_error').mean()))


# In[38]:


subm['Item_Outlet_Sales'] = test_dummy['Item_Outlet_Sales']
subm.to_csv('LinearRegression.csv', index = False)


# Cross-validation score: 1133.7714657463666
# 
# Public Leaderboard Score: 1202.84839849882

# In[39]:


coeff_lin_regr = pd.Series(lin_regr.coef_, pred).sort_values()
coeff_len_regr.plot(kind='bar', title='Model Coefficients')


# We see that some of the coefficients are pretty large so we will try to regulize the model by using Ridge and Lasso regression.

# ## Ridge Regression

# In[40]:


ridge_regr = Ridge(alpha=.1, normalize=True)
ridge_regr.fit(train_dummy[pred], train_dummy['Item_Outlet_Sales'])
test_dummy['Item_Outlet_Sales'] = ridge_regr.predict(test_dummy[pred])


# In[41]:


cv_score = np.sqrt(np.abs(cross_val_score(ridge_regr, train_dummy[pred], train_dummy['Item_Outlet_Sales'], 
                cv = 3, scoring='neg_mean_squared_error').mean()))


# In[42]:


coeff_ridge = pd.Series(ridge_regr.coef_, pred).sort_values()
coeff_ridge.plot(kind='bar', title='Model Coefficients')


# In[ ]:


subm['Item_Outlet_Sales'] = test_dummy['Item_Outlet_Sales']
subm.to_csv('RidgeRegression.csv', index = False)


# Note that the coefficients now look better but even with a very little regularization parameter alpha (0.1) we still get cross-validation score of 1137.3349089796116 which is not much better than the unregularized model. 

# ## Lasso Regression

# In[43]:


lasso_regr = Lasso(alpha=.1, normalize=True)
lasso_regr.fit(train_dummy[pred], train_dummy['Item_Outlet_Sales'])
test_dummy['Item_Outlet_Sales'] = lasso_regr.predict(test_dummy[pred])


# In[44]:


cv_score = np.sqrt(np.abs(cross_val_score(lasso_regr, train_dummy[pred], train_dummy['Item_Outlet_Sales'], 
                cv = 3, scoring='neg_mean_squared_error').mean()))


# In[45]:


coeff_lasso = pd.Series(lasso_regr.coef_, pred).sort_values()
coeff_lasso.plot(kind='bar', title='Model Coefficients')


# In[ ]:


subm['Item_Outlet_Sales'] = test_dummy['Item_Outlet_Sales']
subm.to_csv('LassoRegression.csv', index = False)


# The same happens here. Again with small alpha (0.1) we get a cross-validation score of 1131.8868997418629.

# ## Decision Tree

# It seems that the linear models don't fit the data adequately. Now we will fit a decision tree model.

# In[102]:


dec_tree = DecisionTreeRegressor(max_depth=6, min_samples_leaf=80)
dec_tree.fit(train_dummy[pred], train_dummy['Item_Outlet_Sales'])
test_dummy['Item_Outlet_Sales'] = dec_tree.predict(test_dummy[pred])


# In[103]:


cv_score = np.sqrt(np.abs(cross_val_score(dec_tree, train_dummy[pred], train_dummy['Item_Outlet_Sales'], 
                cv = 3, scoring='neg_mean_squared_error').mean()))


# In[104]:


cv_score


# In[105]:


coeff_dec_tree = pd.Series(dec_tree.feature_importances_, pred).sort_values(ascending=False)
coeff_dec_tree.plot(kind='bar', title='Feature Importances')


# In[106]:


subm['Item_Outlet_Sales'] = test_dummy['Item_Outlet_Sales']
subm.to_csv('DecisionTree.csv', index = False)


# Cross-validation score: 1093.7751907449392
# 
# Public Leaderboard Score: 1159.26221440279

# ## Random Forest

# In[108]:


random_forest = RandomForestRegressor(n_estimators=200,max_depth=5, min_samples_leaf=100,n_jobs=4)
random_forest.fit(train_dummy[pred], train_dummy['Item_Outlet_Sales'])
test_dummy['Item_Outlet_Sales'] = random_forest.predict(test_dummy[pred])


# In[109]:


cv_score = np.sqrt(np.abs(cross_val_score(random_forest, train_dummy[pred], train_dummy['Item_Outlet_Sales'], 
                cv = 3, scoring='neg_mean_squared_error').mean()))


# In[110]:


cv_score


# In[111]:


coeff_random_forest = pd.Series(random_forest.feature_importances_, pred).sort_values(ascending=False)
coeff_random_forest.plot(kind='bar', title='Feature Importances')


# In[112]:


subm['Item_Outlet_Sales'] = test_dummy['Item_Outlet_Sales']
subm.to_csv('RandomForest.csv', index = False)


# Cross-validation score: 1093.4833737190904
# 
# Public Leaderboard Score: 1153.89144127512

# ## XGB

# In[225]:


import xgboost as xgb
xgb = xgb.XGBRegressor(objective ='reg:linear', learning_rate = 0.4, subsample = 0.8, colsample_bytree = 0.6, 
                min_child_weight = 3, max_depth = 4, n_estimators = 10)
xgb.fit(train_dummy[pred], train_dummy['Item_Outlet_Sales'])
test_dummy['Item_Outlet_Sales'] = xgb.predict(test_dummy[pred])


# In[226]:


cv_score = np.sqrt(np.abs(cross_val_score(xgb, train_dummy[pred], train_dummy['Item_Outlet_Sales'], 
                cv = 3, scoring='neg_mean_squared_error').mean()))


# In[227]:


cv_score


# In[228]:


coeff_xgb = pd.Series(xgb.feature_importances_, pred).sort_values(ascending=False)
coeff_xgb.plot(kind='bar', title='Feature Importances')


# In[229]:


subm['Item_Outlet_Sales'] = test_dummy['Item_Outlet_Sales']
subm.to_csv('xgb.csv', index = False)


# Cross-validation score: 1091.031848644131
# 
# Public Leaderboard Score: 1156.92198935604
# 
# 
# Usually xgb performs better than RandomForest so we need to tune the parameters to improve the performance.

# # End Notes

# In all cases we get a cross-validation score lower than the actual public leaderboard score which indicates that we are overfitting the data. We might want to revisit the predictors we use. In addition, parameter tuning is required in order to optimize the results.

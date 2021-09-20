#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

train_data = pd.read_csv("../data/train.csv")
test_data = pd.read_csv("../data/test.csv")


# In[2]:


train_data["Date"] = pd.to_datetime(train_data.Date,format="%Y-%m-%d")
test_data["Date"] = pd.to_datetime(test_data.Date,format="%Y-%m-%d")


# In[ ]:





# In[3]:


# train_data["weekend"] = train_data["weekend"].astype("str")
# train_data["dayofweek"] = train_data["dayofweek"].astype("str")


# In[4]:


train_data.dtypes


# In[5]:


train_data.set_index("ID",inplace=True)
test_data.set_index("ID",inplace=True)

bool_int_map={True:1, False:0}

# #Order column is missing in the input test data. Hence this data will be missing
train_data["Date"] = pd.to_datetime(train_data.Date,format="%Y-%m-%d")
test_data["Date"] = pd.to_datetime(test_data.Date,format="%Y-%m-%d")

# Is the day a weekend?
train_data["dayofweek"]=train_data["Date"].dt.dayofweek
test_data["dayofweek"]=test_data["Date"].dt.dayofweek
train_data["weekend"] = train_data["dayofweek"]>=5
test_data["weekend"] = test_data["dayofweek"]>=5
train_data.drop('dayofweek', inplace=True, axis=1)
test_data.drop('dayofweek', inplace=True, axis=1)

train_data["weekend"] = train_data["weekend"].map(bool_int_map)
test_data["weekend"] = test_data["weekend"].map(bool_int_map)

# train_data["quarter"]=train_data["Date"].dt.quarter
# test_data["quarter"]=test_data["Date"].dt.quarter

# train_data["Date"] = train_data["Date"].dt.date
# test_data["Date"] = test_data["Date"].dt.date
# # holiday_map = {0:False, 1:True}
# train_data["Holiday"] = train_data["Holiday"].map(holiday_map)
# test_data["Holiday"] = test_data["Holiday"].map(holiday_map)

discount_map = {"Yes":1, "No":0}
train_data["Discount"] = train_data["Discount"].map(discount_map)
test_data["Discount"] = test_data["Discount"].map(discount_map)

categ_fts =[ 'Store_Type', 'Location_Type', 'Region_Code']
for ft in categ_fts:
    train_data[ft] = train_data[ft].astype("category")
    test_data[ft] = test_data[ft].astype("category")


# In[6]:


# int_bool_map = {0:False, 1:True}

# train_data["Holiday"] = train_data["Holiday"].map(int_bool_map)
# train_data["Discount"] = train_data["Discount"].map(int_bool_map)
train_data


# In[7]:


import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

num_fts = [x for x in test_data.select_dtypes(include="number").columns]
cat_fts = [x for x in test_data.select_dtypes(include=["bool","category"]).columns if x !="Store_id"]


target="Sales"
features=[x for x in test_data.columns if x not in target]


# In[8]:


features
# train_data["Holiday"] = train_data["Holiday"].map(bool_int_map)
# train_data["Discount"] = train_data["Discount"].map(bool_int_map)


# In[9]:


for ft in cat_fts:
    dummy = pd.get_dummies(train_data[ft], prefix=f"{ft[0]}{ft[ft.find('_')+1]}", drop_first=True)
    train_data = pd.merge(train_data, dummy, left_index=True, right_index=True).drop(ft,axis=1)


# In[10]:


for ft in cat_fts:
    dummy = pd.get_dummies(test_data[ft], prefix=f"{ft[0]}{ft[ft.find('_')+1]}", drop_first=True)
    test_data = pd.merge(test_data, dummy, left_index=True, right_index=True).drop(ft,axis=1)


# In[11]:


features = [x for x in test_data.select_dtypes("number").columns]


# In[12]:


test_data.select_dtypes("number").columns


# In[13]:


dy2 = train_data[(train_data["Sales"]<200000)]
dy = train_data[(train_data["Sales"] <150000)]


# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler


# In[15]:


from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor,BaggingRegressor, VotingRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import HuberRegressor, LinearRegression


# In[16]:


s_scaler = StandardScaler()
# for ft in features:
#     s_scaler.fit(train_data[ft].values.reshape(-1,1))
#     train_data[ft] = s_scaler.transform(train_data[ft].values.reshape(-1,1))
#     test_data[ft] = s_scaler.transform(test_data[ft].values.reshape(-1,1))


# In[17]:


# sns.histplot(train_data)


# In[18]:


train_data


# In[19]:


X_train, X_test, y_train, y_test = train_test_split(train_data[features], train_data[target], test_size = 0.05)
models=[DecisionTreeRegressor, RandomForestRegressor]
for model in models:
    print(model)
    model_instance = model()
    model_instance.fit(X_train, y_train)
    y_hat = model_instance.predict(X_test)
    print(f"test_error: {mse(y_true=y_test, y_pred=y_hat)}")
    print(f"train_error: {mse(y_true=y_train, y_pred=model_instance.predict(X_train))}")
#     X_test[target]=model_instance.predict(X_test)
#     break


# In[20]:


# weekend and weekday info


# <class 'sklearn.tree._classes.DecisionTreeRegressor'>
# test_error: 130746800.6851016
# train_error: 108954910.33079284
# <class 'sklearn.tree._classes.DecisionTreeRegressor'>
# test_error: 108789740.10433578
# train_error: 103783211.1154453
# <class 'sklearn.ensemble._forest.RandomForestRegressor'>
# test_error: 108814290.20272683
# train_error: 103807666.99687739

# In[21]:


train_data.Date


# In[22]:


train_data


# In[23]:


model_instance = DecisionTreeRegressor()
# model_instance = RandomForestRegressor()
model_instance.fit(train_data[features], train_data[target])
test_data[f"{target}"] = model_instance.predict(test_data[features])
test_data[target].to_csv("submission_DTRegressor.csv")
# test_data[target].to_csv("submission_RFRegressor.csv")


# In[24]:


train_data[features]


# In[25]:


import xgboost as xgb
from xgboost import XGBRegressor
regressor = XGBRegressor()
dtrain = xgb.DMatrix(train_data[features])


# In[26]:


model = XGBRegressor(
    max_depth=10,
    n_estimators=1000,
    min_child_weight=0.5, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.1,
    importance_type='gini',
    seed=42)

model.fit(
    X_train, 
    y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, y_train), (X_test, y_test)], 
    verbose=True, 
    early_stopping_rounds = 2)

# rcParams['figure.figsize'] = 12, 4


# In[27]:


# Further optimizing this model resulted in deteoriation of rank on public leader dashboard.
# validation_0-rmse:10214.00488	validation_1-rmse:10396.72754 
# validation_0-rmse:9834.18750	validation_1-rmse:10185.91602
# validation_0-rmse:9854.59570	validation_1-rmse:10168.09473
# Hence reverting back 


# In[28]:


test_data[target] =  model.predict(test_data[features])


# In[29]:


test_data


# In[30]:


# test_data


# In[31]:


test_data[target].to_csv("submission_XGBRegressor.csv")


# In[ ]:





# In[ ]:





# In[ ]:





# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 15:50:23 2017

@author: bowen
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')


""""
Data Preprocessing
""""

# Remove outliers
xlim = [-74.03, -73.77]
ylim = [40.63, 40.85]
dataset = dataset[(dataset.pickup_longitude> xlim[0]) & (dataset.pickup_longitude < xlim[1])]
dataset = dataset[(dataset.dropoff_longitude> xlim[0]) & (dataset.dropoff_longitude < xlim[1])]
dataset = dataset[(dataset.pickup_latitude> ylim[0]) & (dataset.pickup_latitude < ylim[1])]
dataset = dataset[(dataset.dropoff_latitude> ylim[0]) & (dataset.dropoff_latitude < ylim[1])]

dataset.to_csv('train_processed.csv', index = False)

train = pd.read_csv('train_processed.csv')



split = train.sample(frac=0.1, replace=True)

"""
Convert Datetime
"""


train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)
train['dropoff_datetime'] = pd.to_datetime(train.dropoff_datetime)
train['pickup_hour'] = train.pickup_datetime.dt.hour
train['day_of_year'] = train.pickup_datetime.dt.dayofyear
train['day_of_week'] = train.pickup_datetime.dt.dayofweek


"""
Visualization
"""
# Remove outliers
xlim = [-74.03, -73.77]
ylim = [40.63, 40.85]
train = train[(train.pickup_longitude> xlim[0]) & (train.pickup_longitude < xlim[1])]
train = train[(train.dropoff_longitude> xlim[0]) & (train.dropoff_longitude < xlim[1])]
train = train[(train.pickup_latitude> ylim[0]) & (train.pickup_latitude < ylim[1])]
train = train[(train.dropoff_latitude> ylim[0]) & (train.dropoff_latitude < ylim[1])]
# Visualize data points
longitude = list(train.pickup_longitude) + list(train.dropoff_longitude)
latitude = list(train.pickup_latitude) + list(train.dropoff_latitude)
plt.figure(figsize = (10,10))
plt.plot(longitude,latitude,'.', alpha = 0.4, markersize = 0.05)
plt.show()


f,axarr = plt.subplots(ncols=2,nrows=1,figsize=(12,6))
axarr[0].scatter(range(train.shape[0]), np.sort(train.trip_duration.values))
q = train.trip_duration.quantile(0.99)
train = train[train.trip_duration < q]
axarr[1].scatter(range(train.shape[0]), np.sort(train.trip_duration.values))

plt.show()




"""
Calculate Distances
"""


import geopy.distance
def getDistance (pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    return geopy.distance.vincenty( (pickup_lat, pickup_lon), (dropoff_lat, dropoff_lon) ).km


#train['distance'] = getDistance(train.pickup_longitude, train.pickup_latitude,
#                                train.dropoff_longitude, train.dropoff_latitude)

train['trip_distance'] = train.apply(lambda x: getDistance(x['pickup_latitude'], 
                                                       x['pickup_longitude'],
                                                        x['dropoff_latitude'],
                                                        x['dropoff_longitude']
                                                       ), axis = 1 )


"""
K-means clustering
"""

from sklearn.cluster import MiniBatchKMeans

coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values, train[['dropoff_latitude', 'dropoff_longitude']].values))
sample_ind = np.random.permutation(len(coords))[:500000]
kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])

train['pickup_cluster'] = kmeans.predict(train[['pickup_latitude', 'pickup_longitude']])
train['dropoff_cluster'] = kmeans.predict(train[['dropoff_latitude', 'dropoff_longitude']])




"""
Test set processing
"""

test = pd.read_csv('test.csv')
test['pickup_cluster'] = kmeans.predict(test[['pickup_latitude', 'pickup_longitude']])
test['dropoff_cluster'] = kmeans.predict(test[['dropoff_latitude', 'dropoff_longitude']])
test.pickup_datetime=pd.to_datetime(test.pickup_datetime)
test['pickup_hour'] = test.pickup_datetime.dt.hour
test['day_of_year'] = test.pickup_datetime.dt.dayofyear
test['day_of_week'] = test.pickup_datetime.dt.dayofweek
test['trip_distance'] = test.apply(lambda x: getDistance(x['pickup_latitude'], 
                                                       x['pickup_longitude'],
                                                        x['dropoff_latitude'],
                                                        x['dropoff_longitude']
                                                       ), axis = 1 )




"""
Training and test split
"""
y = np.log(train['trip_duration'].values + 1)
from sklearn.cross_validation import train_test_split
features = ['vendor_id',
             'pickup_hour',
             'day_of_year',
             'day_of_week',
             'trip_distance',
             'pickup_cluster',
             'dropoff_cluster']
X_train, X_test, y_train, y_test = train_test_split(train[features], y, test_size = 0.2, random_state = 0)




"""
Random Forest regressor
"""

#random forest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 5, random_state = 0)
regressor.fit(X_train, y_train)
# Predicting random forest results
y_pred = regressor.predict(X_test)



"""
XGBoost 
"""

import xgboost as xgb
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)
dtest = xgb.DMatrix(test[features])
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

# Try different parameters! My favorite is random search :)
xgb_pars = {'min_child_weight': 50, 'eta': 0.3, 'colsample_bytree': 0.3, 'max_depth': 10,
            'subsample': 0.8, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
            'eval_metric': 'rmse', 'objective': 'reg:linear'}
# You could try to train with more epoch
model = xgb.train(xgb_pars, dtrain, 60, watchlist, early_stopping_rounds=50,
                  maximize=False, verbose_eval=10)

y_pred_xgb2 = model.predict(dtest)



"""
LightGBM
"""


import lightgbm as lgb
def lgb_rmsle_score(preds, dtrain):
    labels = np.exp(dtrain.get_label())
    preds = np.exp(preds.clip(min=0))
    return 'rmsle', np.sqrt(np.mean(np.square(np.log1p(preds)-np.log1p(labels)))), False

d_train = lgb.Dataset(X_train, y_train)

lgb_params = {
    'learning_rate': 0.2, # try 0.2
    'max_depth': 8,
    'num_leaves': 55, 
    'objective': 'regression',
    #'metric': {'rmse'},
    'feature_fraction': 0.9,
    'bagging_fraction': 0.5,
    #'bagging_freq': 5,
    'max_bin': 200}       # 1000
cv_result_lgb = lgb.cv(lgb_params,
                       d_train, 
                       num_boost_round=5000, 
                       nfold=3, 
                       feval=lgb_rmsle_score,
                       early_stopping_rounds=50, 
                       verbose_eval=100, 
                       show_stdv=True)
n_rounds = len(cv_result_lgb['rmsle-mean'])
print('num_boost_rounds_lgb=' + str(n_rounds))




def dummy_rmsle_score(preds, y):
    return np.sqrt(np.mean(np.square(np.log1p(np.exp(preds))-np.log1p(np.exp(y)))))

# Train a model
model_lgb = lgb.train(lgb_params, 
                      d_train, 
                      feval=lgb_rmsle_score, 
                      num_boost_round=n_rounds)
# Predict on train
y_train_pred = model_lgb.predict(X_train)
print('RMSLE on train = {}'.format(dummy_rmsle_score(y_train_pred, y_train)))
# Predict on validation
y_valid_pred = model_lgb.predict(X_test)
print('RMSLE on valid = {}'.format(dummy_rmsle_score(y_valid_pred, y_test)))

y_test_pred_lightgbm = model_lgb.predict(test[features])





"""
List feature importance
"""


feature_imp = pd.Series(dict(zip(X_train.columns, model_lgb.feature_importance()))) \
                    .sort_values(ascending=False)
                    
print(feature_imp)


"""
Calculate accuracy
"""

#def rmsle(predicted,real):
#    sum=0.0
#    i= 0
#    for x in range(len(predicted)):
#
#        p = np.log(predicted[x]+1)
#        r = np.log(real[x]+1)
#        sum = sum + (p - r)**2
#    return (sum/len(predicted))**0.5
#
#error = rmsle(y_pred_xgb2, y_test)
#print(error)



"""
Submission
"""

test['trip_duration'] = np.exp(y_pred_xgb2) - 1
out = test[['id','trip_duration']]
out.to_csv('submission.csv', index = False)

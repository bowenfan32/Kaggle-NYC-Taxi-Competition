# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 15:50:23 2017

@author: bowen
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


""""
Training Data Preprocessing
""""

# Importing the dataset
dataset = pd.read_csv('train.csv')

# Add Routes
routes1 = pd.read_csv('fastest_routes_train_part_1.csv')
routes2 = pd.read_csv('fastest_routes_train_part_2.csv')
routes = routes1.append(routes2, ignore_index = True)
routes = routes[['id', 'total_distance', 'total_travel_time', 'number_of_steps']]
dataset = pd.merge(dataset, routes, how = 'left', on = 'id')

## Add weather
#weather = pd.read_csv('weather_daily.csv')
#weather.replace('T', 0.001, inplace=True)
#weather['date'] = pd.to_datetime(weather['date'], dayfirst=True).dt.date
#weather['average temperature'] = weather['average temperature'].astype(np.float64)
#weather['precipitation'] = weather['precipitation'].astype(np.float64)
#weather['snow fall'] = weather['snow fall'].astype(np.float64)
#weather['snow depth'] = weather['snow depth'].astype(np.float64)




dataset.fillna(0, inplace = True)

# Remove outliers
xlim = [-74.03, -73.77]
ylim = [40.63, 40.85]
dataset = dataset[(dataset.pickup_longitude> xlim[0]) & (dataset.pickup_longitude < xlim[1])]
dataset = dataset[(dataset.dropoff_longitude> xlim[0]) & (dataset.dropoff_longitude < xlim[1])]
dataset = dataset[(dataset.pickup_latitude> ylim[0]) & (dataset.pickup_latitude < ylim[1])]
dataset = dataset[(dataset.dropoff_latitude> ylim[0]) & (dataset.dropoff_latitude < ylim[1])]

dataset.to_csv('train_processed.csv', index = False)



"""
Test Data Preprocessing
"""

testset = pd.read_csv('test.csv')

routes = pd.read_csv('fastest_routes_test.csv')
routes = routes[['id', 'total_distance', 'total_travel_time', 'number_of_steps']]
testset = pd.merge(testset, routes, how = 'left', on = 'id')


testset.to_csv('test_processed.csv', index = False)

"""
Convert Datetime
"""

train = pd.read_csv('train_processed.csv')
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


def manhattan_distances(x1, x2, y1, y2):
    return np.abs(x1 - x2) + np.abs(y1 - y2)


#train['distance'] = getDistance(train.pickup_longitude, train.pickup_latitude,
#                                train.dropoff_longitude, train.dropoff_latitude)

train['trip_distance'] = train.apply(lambda x: getDistance(x['pickup_latitude'], 
                                                       x['pickup_longitude'],
                                                        x['dropoff_latitude'],
                                                        x['dropoff_longitude']
                                                       ), axis = 1 )

train['trip_distance_manhattan'] = manhattan_distances(train['pickup_longitude'],
                                                      train['dropoff_longitude'],
                                                     train['pickup_latitude'],
                                                    train['dropoff_latitude'])


"""
Bearing feature
"""


def bearing_array(lat1, lng1, lat2, lng2):
    AVG_EARTH_RADIUS = 6371  # in km
    lng_delta_rad = np.radians(lng2 - lng1)
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    y = np.sin(lng_delta_rad) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(lng_delta_rad)
    return np.degrees(np.arctan2(y, x))

train['direction'] = bearing_array(train['pickup_latitude'].values, train['pickup_longitude'].values, 
                                          train['dropoff_latitude'].values, train['dropoff_longitude'].values)



"""
K-means clustering
"""

from sklearn.cluster import MiniBatchKMeans

coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values, train[['dropoff_latitude', 'dropoff_longitude']].values))
sample_ind = np.random.permutation(len(coords))[:500000]
kmeans = MiniBatchKMeans(n_clusters=100, batch_size=10000).fit(coords[sample_ind])

train['pickup_cluster'] = kmeans.predict(train[['pickup_latitude', 'pickup_longitude']])
train['dropoff_cluster'] = kmeans.predict(train[['dropoff_latitude', 'dropoff_longitude']])

train.to_csv('train_processed_converted', index=False)


"""
Test set processing
"""

test = pd.read_csv('test_processed.csv')
test['pickup_cluster'] = kmeans.predict(test[['pickup_latitude', 'pickup_longitude']])
test['dropoff_cluster'] = kmeans.predict(test[['dropoff_latitude', 'dropoff_longitude']])
test.pickup_datetime=pd.to_datetime(test.pickup_datetime)
test['pickup_hour'] = test.pickup_datetime.dt.hour
test['day_of_year'] = test.pickup_datetime.dt.dayofyear
test['day_of_week'] = test.pickup_datetime.dt.dayofweek
test['direction'] = bearing_array(test['pickup_latitude'].values, test['pickup_longitude'].values, 
                                         test['dropoff_latitude'].values, test['dropoff_longitude'].values)

test['trip_distance'] = test.apply(lambda x: getDistance(x['pickup_latitude'], 
                                                       x['pickup_longitude'],
                                                        x['dropoff_latitude'],
                                                        x['dropoff_longitude']
                                                       ), axis = 1 )
test['trip_distance_manhattan'] = manhattan_distances(test['pickup_longitude'],
                                                      test['dropoff_longitude'],
                                                     test['pickup_latitude'],
                                                    test['dropoff_latitude'])

test.to_csv('test_processed_converted', index=False)

"""
Training and test split
"""
y = np.log(train['trip_duration'].values + 1)
from sklearn.cross_validation import train_test_split
features = ['vendor_id',
             'passenger_count',
             'pickup_hour',
             'day_of_year',
             'day_of_week',
             'trip_distance',
             'trip_distance_manhattan',
             'pickup_cluster',
             'dropoff_cluster',
             'total_distance',
             'total_travel_time',
             'number_of_steps',
             'direction'
#             'total_distance_2',
#             'total_travel_time_2',
#             'number_of_steps_2'
             ]
X_train, X_test, y_train, y_test = train_test_split(train[features], y, test_size = 0.2, random_state = 0)




"""
XGBoost 
"""

import xgboost as xgb
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_test, label=y_test)
dtest = xgb.DMatrix(test[features])
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]

# Try different parameters! My favorite is random search :)
xgb_pars = {'min_child_weight': 10, 'eta': 0.05, 'colsample_bytree': 0.8, 'max_depth': 10,
            'subsample': 0.8, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
            'eval_metric': 'rmse', 'objective': 'reg:linear'}
# You could try to train with more epoch
model = xgb.train(xgb_pars, dtrain, 6000, watchlist, early_stopping_rounds=50,
                  maximize=False, verbose_eval=10)

y_pred_xgb2 = model.predict(dtest)






#from sklearn.grid_search import GridSearchCV
#cv_params = {'max_depth': [3,5,7], 'min_child_weight': [1,3,5]}
#ind_params = {'learning_rate': 0.1, 'n_estimators': 1000, 'seed':0, 'subsample': 0.8, 'colsample_bytree': 0.8, 
#             'objective': 'binary:logistic'}
#optimized_GBM = GridSearchCV(xgb.XGBClassifier(**ind_params), 
#                            cv_params, 
#                             scoring = 'accuracy', cv = 5, n_jobs = -1) 
#
#optimized_GBM.fit(X_train, y_train)
#print (optimized_GBM.grid_scores_)



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
    'learning_rate': 0.05, # try 0.2
    'max_depth': 8,
    'num_leaves': 80, 
    'objective': 'regression',
    #'metric': {'rmse'},
    'feature_fraction': 0.9,
    'bagging_fraction': 0.5,
    #'bagging_freq': 5,
    'max_bin': 1000}       # 1000
cv_result_lgb = lgb.cv(lgb_params,
                       d_train, 
                       num_boost_round=50000, 
                       nfold=3, 
                       feval=lgb_rmsle_score,
                       early_stopping_rounds=50, 
                       verbose_eval=50, 
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

test['trip_duration'] = np.exp(y_test_pred_lightgbm) - 1
out = test[['id','trip_duration']]
out.to_csv('submission.csv', index = False)
#out.drop_duplicates(subset = ['id'], keep = 'first').to_csv('submission.csv', index = False)

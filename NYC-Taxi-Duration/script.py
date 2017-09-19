# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 15:50:23 2017

@author: bowen
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


"""
Training Data Preprocessing
"""

# Importing the dataset
dataset = pd.read_csv('train.csv')

# Add Routes
routes1 = pd.read_csv('fastest_routes_train_part_1.csv')
routes2 = pd.read_csv('fastest_routes_train_part_2.csv')
routes = routes1.append(routes2, ignore_index = True)
routes = routes[['id', 'total_distance', 'total_travel_time', 'number_of_steps']]
dataset = pd.merge(dataset, routes, how = 'left', on = 'id')



# Remove location outliers
xlim = [-74.03, -73.77]
ylim = [40.63, 40.85]
dataset = dataset[(dataset.pickup_longitude> xlim[0]) & (dataset.pickup_longitude < xlim[1])]
dataset = dataset[(dataset.dropoff_longitude> xlim[0]) & (dataset.dropoff_longitude < xlim[1])]
dataset = dataset[(dataset.pickup_latitude> ylim[0]) & (dataset.pickup_latitude < ylim[1])]
dataset = dataset[(dataset.dropoff_latitude> ylim[0]) & (dataset.dropoff_latitude < ylim[1])]

#Remove trip duration outliers
m = np.mean(dataset['trip_duration'])
s = np.std(dataset['trip_duration'])
dataset = dataset[dataset['trip_duration'] <= m + 2*s]
dataset = dataset[dataset['trip_duration'] >= m - 2*s]


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
train['pickup_hour'] = train.pickup_datetime.dt.hour
train['day_of_year'] = train.pickup_datetime.dt.dayofyear
train['day_of_week'] = train.pickup_datetime.dt.dayofweek
train['minute_of_hour'] = train.pickup_datetime.dt.minute



"""
Calculate Distances
"""


def haversine_distance(lat1, lng1, lat2, lng2):
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    AVG_EARTH_RADIUS = 6371  # in km
    lat = lat2 - lat1
    lng = lng2 - lng1
    d = np.sin(lat * 0.5) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(lng * 0.5) ** 2
    h = 2 * AVG_EARTH_RADIUS * np.arcsin(np.sqrt(d))
    return h

def manhattan_distance(lat1, lng1, lat2, lng2):
    a = haversine_distance(lat1, lng1, lat1, lng2)
    b = haversine_distance(lat1, lng1, lat2, lng1)
    return a + b


train['trip_distance'] = haversine_distance(train['pickup_latitude'], 
                                                   train['pickup_longitude'],
                                                    train['dropoff_latitude'],
                                                    train['dropoff_longitude'])

train['trip_distance_manhattan'] = manhattan_distance(train['pickup_latitude'],
                                                      train['pickup_longitude'],
                                                     train['dropoff_latitude'],
                                                    train['dropoff_longitude'])


"""
Bearing feature
"""


def bearing_array(lat1, lng1, lat2, lng2):
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



"""
PCA features
"""
from sklearn.decomposition import PCA
coords = np.vstack((train[['pickup_latitude', 'pickup_longitude']].values,
                    train[['dropoff_latitude', 'dropoff_longitude']].values))

pca = PCA().fit(coords)

train['pickup_pca0'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 0]
train['pickup_pca1'] = pca.transform(train[['pickup_latitude', 'pickup_longitude']])[:, 1]
train['dropoff_pca0'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
train['dropoff_pca1'] = pca.transform(train[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
train['pca_manhattan'] = np.abs(train['dropoff_pca1'] - train['pickup_pca1']) + np.abs(train['dropoff_pca0'] - train['pickup_pca0'])



"""
Extra features
"""

train['pickup_latitude_round3'] = np.round(train['pickup_latitude'], 3)
train['pickup_longitude_round3'] = np.round(train['pickup_longitude'], 3)
train['dropoff_latitude_round3'] = np.round(train['dropoff_latitude'], 3)
train['dropoff_longitude_round3'] = np.round(train['dropoff_longitude'], 3)
train['center_latitude'] = (train['pickup_latitude_round3'].values + train['dropoff_latitude_round3'].values) / 2
train['center_longitude'] = (train['pickup_longitude_round3'].values + train['dropoff_longitude_round3'].values) / 2



"""
Add and merge weather data
"""

## Add weather
#weather = pd.read_csv('weather_daily.csv')
#weather.replace('T', 0.001, inplace=True)
#weather['date'] = pd.to_datetime(weather['date'], dayfirst=True).dt.date
#weather['average temperature'] = weather['average temperature'].astype(np.float64)
#weather['precipitation'] = weather['precipitation'].astype(np.float64)
#weather['snow fall'] = weather['snow fall'].astype(np.float64)
#weather['snow depth'] = weather['snow depth'].astype(np.float64)
weather_hour = pd.read_csv('weather_hourly.csv')
weather_hour['Datetime'] = pd.to_datetime(weather_hour['pickup_datetime'], dayfirst=True)
weather_hour['date'] = weather_hour.Datetime.dt.date
weather_hour['pickup_hour'] = weather_hour.Datetime.dt.hour
weather_hour['pickup_hour'] = weather_hour.pickup_hour.astype(np.int8)
weather_hour['fog'] = weather_hour.fog.astype(np.int8)
weather_hour = weather_hour[['date', 'pickup_hour', 'tempm', 'dewptm', 'hum', 'wspdm', 
                             'wdird', 'vism', 'pressurei', 'fog']]
train.fillna(0, inplace = True)

train['date'] = pd.to_datetime(train.pickup_datetime).dt.date # adding date column
train = pd.merge(left=train, right=weather_hour.drop_duplicates(subset=['date', 'pickup_hour']), 
                  on=['date', 'pickup_hour'], how='left')








train.to_csv('train_processed_converted', index=False)

train = pd.read_csv('train_processed_converted')



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
test['minute_of_hour'] = test.pickup_datetime.dt.minute
test['direction'] = bearing_array(test['pickup_latitude'].values, test['pickup_longitude'].values, 
                                         test['dropoff_latitude'].values, test['dropoff_longitude'].values)

test['trip_distance'] = haversine_distance(test['pickup_latitude'], 
                                                   test['pickup_longitude'],
                                                    test['dropoff_latitude'],
                                                    test['dropoff_longitude'])

test['trip_distance_manhattan'] = manhattan_distance(test['pickup_latitude'],
                                                      test['pickup_longitude'],
                                                     test['dropoff_latitude'],
                                                    test['dropoff_longitude'])

coords = np.vstack((test[['pickup_latitude', 'pickup_longitude']].values,
                    test[['dropoff_latitude', 'dropoff_longitude']].values))
pca = PCA().fit(coords)
test['pickup_pca0'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 0]
test['pickup_pca1'] = pca.transform(test[['pickup_latitude', 'pickup_longitude']])[:, 1]
test['dropoff_pca0'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 0]
test['dropoff_pca1'] = pca.transform(test[['dropoff_latitude', 'dropoff_longitude']])[:, 1]
test['pca_manhattan'] = np.abs(test['dropoff_pca1'] - test['pickup_pca1']) + np.abs(test['dropoff_pca0'] - test['pickup_pca0'])

test['pickup_latitude_round3'] = np.round(test['pickup_latitude'], 3)
test['pickup_longitude_round3'] = np.round(test['pickup_longitude'], 3)
test['dropoff_latitude_round3'] = np.round(test['dropoff_latitude'], 3)
test['dropoff_longitude_round3'] = np.round(test['dropoff_longitude'], 3)
test['center_latitude'] = (test['pickup_latitude_round3'].values + test['dropoff_latitude_round3'].values) / 2
test['center_longitude'] = (test['pickup_longitude_round3'].values + test['dropoff_longitude_round3'].values) / 2


test['date'] = pd.to_datetime(test.pickup_datetime).dt.date # adding date column
test = pd.merge(left=test, right=weather_hour.drop_duplicates(subset=['date', 'pickup_hour']), 
                 on=['date', 'pickup_hour'], how='left')
test.fillna(0, inplace = True)




test.to_csv('test_processed_converted', index=False)

test = pd.read_csv('test_processed_converted')


"""
Training and test split
"""
y = np.log(train['trip_duration'].values + 1)
from sklearn.cross_validation import train_test_split
features = [
            'vendor_id',
             'passenger_count',
             'pickup_hour',
             'day_of_year',
             'day_of_week',
             'minute_of_hour',
             'trip_distance',
             'trip_distance_manhattan',
             'pickup_cluster',
             'dropoff_cluster',
             'total_distance',
             'total_travel_time',
             'number_of_steps',
             'direction',
             'pickup_pca0',
             'pickup_pca1',
             'dropoff_pca0',
             'dropoff_pca1',
             'pca_manhattan',
#             'pickup_latitude_round3',
#             'pickup_longitude_round3',
#             'dropoff_latitude_round3',
#             'dropoff_longitude_round3',
             'center_latitude',
             'center_longitude',
             'tempm', 'dewptm', 'hum', 'wspdm', 'wdird', 'pressurei'
#             'vism','fog'
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

xgb_pars = {'min_child_weight': 5, 'eta': 0.05, 'colsample_bytree': 0.8, 'max_depth': 10,
            'subsample': 0.8, 'lambda': 1., 'nthread': -1, 'booster' : 'gbtree', 'silent': 1,
            'eval_metric': 'rmse', 'objective': 'reg:linear'}
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
    'max_depth': 10,
    'num_leaves':70, 
    'objective': 'regression',
#    'seed': 2017,
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
                      verbose_eval=True,
                      num_boost_round=n_rounds)
# Predict on train
y_train_pred = model_lgb.predict(X_train)
print('RMSLE on train = {}'.format(dummy_rmsle_score(y_train_pred, y_train)))
# Predict on validation
y_valid_pred = model_lgb.predict(X_test)
print('RMSLE on valid = {}'.format(dummy_rmsle_score(y_valid_pred, y_test)))

y_test_pred_lightgbm = model_lgb.predict(test[features])



"""
Grid Search
"""

from sklearn.model_selection import GridSearchCV
from lightgbm.sklearn import LGBMRegressor
estimator = LGBMRegressor(
        num_leaves = 80, # cv调节50是最优值
        max_depth = 8,
        learning_rate =0.3, 
        n_estimators = 1000, 
        objective = 'regression', 
        min_child_weight = 1, 
#        subsample = 0.8,
#        colsample_bytree=0.8,
        nthread = 7,
    )


def print_best_score(gsearch,param_test):
     # 输出best score
    print("Best score: %0.3f" % gsearch.best_score_)
    print("Best parameters set:")
    # 输出最佳的分类器到底使用了怎样的参数
    best_parameters = gsearch.best_estimator_.get_params()
    for param_name in sorted(param_test.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

# Create parameters to search
gridParams = {
    'learning_rate': [0.3],
#    'n_estimators': [8,24,48],
    'num_leaves': range(50, 100, 10),
#    'boosting_type' : ['gbdt'],
#    'objective' : ['binary'],
#    'seed' : [500],
#    'colsample_bytree' : [0.65, 0.75, 0.8],
#    'subsample' : [0.7,0.75],
    'reg_alpha': [1,2,6],
    'reg_lambda': [1,2,6],
    }
# Create the grid
grid = GridSearchCV(estimator, gridParams, verbose=10, cv=4, n_jobs=-1)
# Run the grid
grid.fit(X_train, y_train)
print_best_score(grid,gridParams)




"""
List feature importance
"""


feature_imp = pd.Series(dict(zip(X_train.columns, model_lgb.feature_importance()))) \
                    .sort_values(ascending=False)
                    
print(feature_imp)



"""
Submission
"""

test['trip_duration'] = np.exp(y_test_pred_lightgbm) - 1
out = test[['id','trip_duration']]
out.to_csv('submission.csv', index = False)








"""
Visualization
"""
# Remove location outliers
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




# Visualize Clusters
city_long_border = (-74.03, -73.75)
city_lat_border = (40.63, 40.85)
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.scatter(train.pickup_longitude.values[:], train.pickup_latitude.values[:], s=1, lw=0,
           c=train.pickup_cluster[:].values, cmap='tab20', alpha=0.2)
ax.set_xlim(city_long_border)
ax.set_ylim(city_lat_border)
ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
plt.show()





fig, ax = plt.subplots(ncols=1, nrows=1)
#ax.plot(train.groupby('pickup_hour').pickup_hour.count(), 'bo-', lw=2, alpha=0.7)
#ax.plot(train.groupby('day_of_week').day_of_week.count(), 'go-', lw=2, alpha=0.7)
ax.plot(train.groupby('day_of_year').day_of_year.count(), 'ro-', lw=2, alpha=0.7)
ax.set_xlabel('Day of the year')
ax.set_ylabel('No. of trips')
fig.suptitle('Number of trips in a year')
plt.show()

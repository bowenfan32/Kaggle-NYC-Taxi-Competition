# -*- coding: utf-8 -*-

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('train.csv')


# Convert pickup date to days of week
from datetime import date
dates = dataset.iloc[:, 2].values
daysOfWeek = []
monthOfYear = []
for i in dates:
    dateObject = date(*map(int, i[:10].split('-')))
    daysOfWeek.append( dateObject.strftime("%A") )
    monthOfYear.append( dateObject.strftime("%B") )
    

# Convert pickup and dropoff coordinate to distances
pickup_lon = dataset.iloc[:, 5].values
pickup_lat = dataset.iloc[:, 6].values
dropoff_lon = dataset.iloc[:, 7].values
dropoff_lat = dataset.iloc[:, 8].values

import geopy.distance
i = 0
distances = []
while i < len(pickup_lon):
    distances.append(geopy.distance.vincenty( (pickup_lat[i], pickup_lon[i]), (dropoff_lat[i], dropoff_lon[i]) ).km)  
    i += 1
    
# Convert time to time of the day
timeOfDay = []
for i in dates:
    time = int(i[11:13])
#    timeOfDay.append(time)
    if (time <= 6):
        timeOfDay.append('Midnight')
    elif (6 < time <= 12):
        timeOfDay.append('Morning')
    elif (12 < time <= 18):
        timeOfDay.append('Afternoon')
    elif (18 < time < 24):
        timeOfDay.append('Night')

# Concatenate generated columns
dfMonths = pd.DataFrame({'Month of Year': monthOfYear})
dfDays = pd.DataFrame({'Days of Week': daysOfWeek})
dfTimes = pd.DataFrame({'Time of the Day': timeOfDay})
dfDistances = pd.DataFrame({'Distances': distances})

result = pd.concat([dataset, dfMonths, dfDays, dfTimes, dfDistances], axis = 1)

result.to_csv('result.csv')

# Encoding categorical data
X = result.iloc[:, [4,11,12,13,14]].values
y = result.iloc[:, 10].values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()


for i in range(4):
    X[:, i] = labelencoder.fit_transform(X[:, i])
#onehotencoder = OneHotEncoder(categorical_features = all)
#X = onehotencoder.fit_transform(X).toarray()


# Splitting the dataset into the Training set and Test set

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting Multiple Linear Regression to the Training set
# from sklearn.linear_model import LinearRegression
# regressorLinear = LinearRegression()
# regressorLinear.fit(X_train, y_train)

# Random Forest
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)




    









# Actual test file prediction
testset = pd.read_csv('test.csv')

dates_test = testset.iloc[:, 2].values
daysOfWeek_test = []
monthOfYear_test = []
for i in dates_test:
    dateObject_test = date(*map(int, i[:10].split('-')))
    daysOfWeek_test.append( dateObject_test.strftime("%A") )
    monthOfYear_test.append( dateObject_test.strftime("%B") )

pickup_lon = testset.iloc[:, 4].values
pickup_lat = testset.iloc[:, 5].values
dropoff_lon = testset.iloc[:, 6].values
dropoff_lat = testset.iloc[:, 7].values

i = 0
distances_test = []
while i < len(pickup_lon):
    distances_test.append(geopy.distance.vincenty( (pickup_lat[i], pickup_lon[i]), (dropoff_lat[i], dropoff_lon[i]) ).km)  
    i += 1
    
timeOfDay_test = []
for i in dates_test:
    time = int(i[11:13])
    timeOfDay_test.append(time)
    if (time <= 6):
        timeOfDay_test.append('Midnight')
    elif (6 < time <= 12):
        timeOfDay_test.append('Morning')
    elif (12< time <= 18):
        timeOfDay_test.append('Afternoon')
    elif (18 < time <= 24):
        timeOfDay_test.append('Night')

dfMonths_test = pd.DataFrame({'Month of Year': monthOfYear_test})
dfDays_test = pd.DataFrame({'Days of Week': daysOfWeek_test})
dfTimes_test = pd.DataFrame({'Time of the Day': timeOfDay_test})
dfDistances_test = pd.DataFrame({'Distances': distances_test})

result_test = pd.concat([testset, dfMonths_test, dfDays_test, dfTimes_test, dfDistances_test], axis = 1)
    
result_test.to_csv('result_test.csv', index=False)

X = result_test.iloc[:, [3,9,10,11,12]].values
labelencoder = LabelEncoder()
for i in range(4):
    X[:, i] = labelencoder.fit_transform(X[:, i])
# onehotencoder = OneHotEncoder(categorical_features = all)
# X = onehotencoder.fit_transform(X).toarray()
dfx = pd.DataFrame(X)

y_pred_test = regressor.predict(X)
dfTripDuration = pd.DataFrame({'trip_duration': y_pred_test})
taxi_id = testset.iloc[:, 0].values
dfId = pd.DataFrame({'id': taxi_id})
submission = pd.concat([dfId, dfTripDuration], axis = 1)

submission.to_csv('submission.csv', index = False)

    
    


    

    
    
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 10:57:03 2019

@author: markus
"""

###############################################
### Winning a Kaggle Competition in Python ####
###############################################


# Working on New York city taxi fare prediction

################
## CHAPTER 1  ##
################

# Import pandas
import pandas as pd

# Read train data
train = pd.read_csv('train.csv')

# Look at the shape of the data
print('Train shape:', train.shape) 

# Look at the head of the data
print(train.head())

#  Congratulations, you've gotten started with your first Kaggle dataset! It contains 15,500 daily observations of the sales data.

#-------

# Explore test data
# Having looked at the train data, let's explore the test data in the "Store Item Demand Forecasting Challenge". Remember, that the test dataset generally contains one column less than the train one.
# This column, together with the output format, is presented in the sample submission file. Before making any progress in the competition, you should get familiar with the expected output.
# That is why, let's look at the columns of the test dataset and compare it to the train columns. Additionally, let's explore the format of the sample submission. The train DataFrame is available in your workspace.

# Read test data
test = pd.read_csv('test.csv')

# Print train and test columns
print('Train columns:', train.columns.tolist())
print('Test columns:', test.columns.tolist())

# Read sample submission
sample_submission = pd.read_csv('sample_submission.csv')

# Look at the head of sample submission
print(sample_submission.head())

# The sample submission file consists of two columns: id of the observation and sales column for your predictions. 
# Kaggle will evaluate your predictions on the true sales data for the corresponding id. 
# So, it’s important to keep track of the predictions by id before submitting them. 
# Let’s jump in the next lesson to see how to prepare a submission file!


#######
#### Prepare your first submission
#######

# Determine a problem type
# You will keep working on the Store Item Demand Forecasting Challenge. 
# Recall that you are given a history of store-item sales data, and asked to predict 3 months of the future sales.
# Before building a model, you should determine the problem type you are addressing. 
# The goal of this exercise is to look at the distribution of the target variable, and select the correct problem type you will be building a model for.
# The train DataFrame is already available in your workspace. It has the target variable column called "sales". Also, matplotlib.pyplot is already imported as plt.

import matlotlib.pyplot as plt
train.sales.hist()
plt.show()  # to see, if variable is continous (regression problem)

# Train a simple model
#As you determined, you are dealing with a regression problem. So, now you're ready to build a model for a subsequent submission. But, instead of building a Linear Regression model as in the slides, let's build a Random Forest model.
#You will use the RandomForestRegressor class from the scikit-learn library.
#Your objective is to train a Random Forest model with default parameters on the "store" and "item" features.

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Read train data
train = pd.read_csv('train.csv')

# Create Random Forest object
rf = RandomForestRegressor()

# Train a model
rf.fit(X=train[["store","item"]], y = train["sales"])


### Prepare a submission
# You've already built a model on the training data from the Kaggle Store Item Demand Forecasting Challenge. Now, it's time to make predictions on the test data and create a submission file in the specified format.
# Your goal is to read the test data, make predictions, and save these in the format specified in the "sample_submission.csv" file. The rf object you created in the previous exercise is available in your workspace.
# Note that starting from now and for the rest of the course, pandas library will be always imported for you and could be accessed as pd.

# Read test and sample submission data
test = pd.read_csv('test.csv')
sample_sub = pd.read_csv('sample_submission.csv')

# Show head of sample_submission
print(sample_sub.head())

# Get predictions for the test set
test['sales'] = rf.predict(test[['store', 'item']])

# Write test predictions in the sample_submission format
test[["id","sales"]].to_csv("kaggle_submission.csv", index=False)

######
### Public vs Private leaderboar
######

# Competition metric: {AUC (class), F1 (class), LogLoss (class), MAE (reg), MSE (reg), MAPK (ranking), MAP@K (ranking) }  .. pozn. MAPK = mean average precision at K

# Treba davat pozor na overfitting medzi public a private leaderboard

# Train XGBoost models
# Every Machine Learning method could potentially overfit. You will see it on this example with XGBoost. Again, you are working with the Store Item Demand Forecasting Challenge. The train DataFrame is available in your workspace.

# Firstly, let's train multiple XGBoost models with different sets of hyperparameters using XGBoost's learning API. The single hyperparameter you will change is:

#max_depth - maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.

import xgboost as xgb

# Create DMatrix on train data
dtrain = xgb.DMatrix(data=train[['store', 'item']],
                     label=train['sales'])



# Try max_depth = {2,8,15}

# Define xgboost parameters
params = {'objective': 'reg:linear',
          'max_depth': 2,
          'silent': 1}

# Train xgboost model
xg_depth_2 = xgb.train(params = params, dtrain = dtrain)


# Define xgboost parameters
params = {'objective': 'reg:linear',
          'max_depth': 8,
          'silent': 1}

# Train xgboost model
xg_depth_8 = xgb.train(params=params, dtrain=dtrain)

# Define xgboost parameters
params = {'objective': 'reg:linear',
          'max_depth': 15,
          'silent': 1}

# Train xgboost model
xg_depth_15 = xgb.train(params=params, dtrain=dtrain)


# Explore overfitting XGBoost
# Having trained 3 XGBoost models with different maximum depths, you will now evaluate their quality. For this purpose, you will measure the quality of each model on both the train data and the test data. As you know by now, the train data is the data models have been trained on. The test data is the next month sales data that models have never seen before.
# The goal of this exercise is to determine whether any of the models trained is overfitting. To measure the quality of the models you will use Mean Squared Error (MSE) available in sklearn.metrics.

# train and test DataFrames together with 3 models trained (xg_depth_2, xg_depth_8, xg_depth_15) are available in your workspace.

from sklearn.metrics import mean_squared_error

dtrain = xgb.DMatrix(data=train[['store', 'item']])
dtest = xgb.DMatrix(data=test[['store', 'item']])

# For each of 3 trained models
for model in [xg_depth_2, xg_depth_8, xg_depth_15]:
    # Make predictions
    train_pred = model.predict(dtrain)     
    test_pred = model.predict(dtest)          
    
    # Compute metrics
    mse_train = mean_squared_error(train['sales'], train_pred)                  
    mse_test = mean_squared_error(test['sales'], test_pred)
    print('MSE Train: {:.3f}. MSE Test: {:.3f}'.format(mse_train, mse_test))


# MSE Train: 631.275. MSE Test: 558.522
# MSE Train: 183.771. MSE Test: 337.337
# MSE Train: 134.984. MSE Test: 355.534    

# So, you see that the third model with depth 15 is already overfitting. 
# It has considerably lower train error compared to the second model, however test error is higher. 
# Be aware of overfitting and move on to the next chapter to know how to beat it!

###########
# Chapter 2 #
#############
    
 # Solution workflow:
    #   understanding the problem -> EDA -> LOCAL VALIDATION -> MODELIN (loop)

# Metrics: from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error
    

### Define a competition metric
#Competition metric is used by Kaggle to evaluate your submissions. Moreover, you also need to measure the performance of different models on a local validation set.

#For now, your goal is to manually develop a couple of competition metrics in case if they are not available in sklearn.metrics.

#In particular, you will define:

## Mean Squared Error (MSE) for the regression problem:
# MSE=1N∑i=1N(yi−y^i)2
    
## Logarithmic Loss (LogLoss) for the binary classification problem:
#LogLoss=−1N∑i=1N(yilogpi+(1−yi)log(1−pi))
    
import numpy as np

# Import MSE from sklearn
from sklearn.metrics import mean_squared_error

# Define your own MSE function
def own_mse(y_true, y_pred):
  	# Find squared differences
    squares = np.power(y_true - y_pred, 2)
    # Find mean over all observations
    err = np.mean(squares)
    return err

print('Sklearn MSE: {:.5f}. '.format(mean_squared_error(y_regression_true, y_regression_pred)))
print('Your MSE: {:.5f}. '.format(own_mse(y_regression_true, y_regression_pred)))


import numpy as np

# Import log_loss from sklearn
from sklearn.metrics import log_loss

# Define your own LogLoss function
def own_logloss(y_true, prob_pred):
  	# Find loss for each observation
    terms = y_true * np.log(prob_pred) + (1 - y_true) * np.log(1 - prob_pred)
    # Find mean over all observations
    err = np.mean(terms) 
    return -err

print('Sklearn LogLoss: {:.5f}'.format(log_loss(y_classification_true, y_classification_pred)))
print('Your LogLoss: {:.5f}'.format(own_logloss(y_classification_true, y_classification_pred)))

## EDA statistics

#As mentioned in the slides, you'll work with New York City taxi fare prediction data. You'll start with finding some basic statistics about the data. Then you'll move forward to plot some dependencies and generate hypotheses on them.
#The train and test DataFrames are already available in your workspace.

# Shapes of train and test data
print('Train shape:', train.shape)
print('Test shape:', test.shape)

# Train head
print(train.head())

# Describe the target variable
print(train.fare_amount.describe())

# Train distribution of passengers within rides
print(train.passenger_count.value_counts())


### EDA plots I

# After generating a couple of basic statistics, it's time to come up with and validate some ideas about the data dependencies. Again, the train DataFrame from the taxi competition is already available in your workspace.

# To begin with, let's make a scatterplot plotting the relationship between the fare amount and the distance of the ride. Intuitively, the longer the ride, the higher its price.
# To get the distance in kilometers between two geo-coordinates, you will use Haversine distance. Its calculation is available with the haversine_distance() function defined for you. The function expects train DataFrame as input.

plt.style.use('ggplot')

# https://en.wikipedia.org/wiki/Haversine_formula

import inspect
print(inspect.getsource(haversine_distance))
def haversine_distance(train):
    
    data = [train]
    lat1, long1, lat2, long2 = 'pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude'
    
    for i in data:
        R = 6371  #radius of earth in kilometers
        #R = 3959 #radius of earth in miles
        phi1 = np.radians(i[lat1])
        phi2 = np.radians(i[lat2])
    
        delta_phi = np.radians(i[lat2]-i[lat1])
        delta_lambda = np.radians(i[long2]-i[long1])
    
        #a = sin²((φB - φA)/2) + cos φA . cos φB . sin²((λB - λA)/2)
        a = np.sin(delta_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2
    
        #c = 2 * atan2( √a, √(1−a) )
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
        #d = R*c
        d = (R * c) #in kilometers
        
    return d

# Calculate the ride distance
train['distance_km'] = haversine_distance(train)

# Draw a scatterplot
plt.scatter(x = train["fare_amount"],y = train["distance_km"], alpha=0.5)
plt.xlabel('Fare amount')
plt.ylabel('Distance, km')
plt.title('Fare amount based on the distance')

# Limit on the distance
plt.ylim(0, 50)
plt.show()


### EDA plots II
# Another idea that comes to mind is that the price of a ride could change during the day.
# Your goal is to plot the median fare amount for each hour of the day as a simple line plot. The hour feature is calculated for you. Don't worry if you do not know how to work with the date features. We will explore them in the chapter on Feature Engineering.

# Create hour feature
train['pickup_datetime'] = pd.to_datetime(train.pickup_datetime)
train['hour'] = train.pickup_datetime.dt.hour

# Find median fare_amount for each hour
hour_price = train.groupby("hour", as_index=False)["fare_amount"].median()

print(hour_price)
# Plot the line plot
plt.plot(hour_price.hour,hour_price.fare_amount, marker='o')
plt.xlabel('Hour of the day')
plt.ylabel('Fare amount')
plt.title('Fare amount based on day time')
plt.xticks(range(24))
plt.show()

# Great! We see that prices are a bit higher during the night. 
# It is a good indicator that we should include the "hour" feature in the final model, or at least add a binary feature "is_night". 
# Move on to the next lesson to learn how to check whether new features are useful for the model or not!



## K-fold cross-validation

# You will start by getting hands-on experience in the most commonly used K-fold cross-validation.
# The data you'll be working with is from "Two sigma connect: rental listing inquiries". The competition problem is a multiclass classification of the rental listings into 3 classes: low interest, medium interest and high interest. For faster performance, you will work with a subsample consisting of 1,000 observations.
# You need to implement a K-fold validation strategy and look at the sizes of each fold obtained. train DataFrame is already available in your workspace.

# Import KFold
from sklearn.model_selection import KFold

# Create a KFold object
kf = KFold(n_splits=3, shuffle=True, random_state=123)

# Loop through each split
fold = 0
for train_index, test_index in kf.split(train):
    # Obtain training and testing folds
    cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]
    print('Fold: {}'.format(fold))
    print('CV train shape: {}'.format(cv_train.shape))
    print('Medium interest listings in CV train: {}\n'.format(sum(cv_train.interest_level == 'medium')))
    fold += 1

# So, we see that the number of observations in each fold is almost uniform. 
# It means that we've just splitted the train data into 3 equal folds. 
# However, if we look at the number of medium-interest listings, it's varying from 162 to 175 from one fold to another. 
# To make them uniform among the folds, let's use Stratified K-fold!    
    
### Stratified K-fold

# As you've just noticed, you have a pretty different target variable distribution among the folds due to the random splits. It's not crucial for this particular competition, but could be an issue for the classification competitions with the highly imbalanced target variable.
# To overcome this, let's implement the stratified K-fold strategy with the stratification on the target variable.
    
# Import StratifiedKFold
from sklearn.model_selection import StratifiedKFold

# Create a StratifiedKFold object
str_kf = StratifiedKFold(n_splits = 3, shuffle = True, random_state=123)

# Loop through each split
fold = 0
for train_index, test_index in str_kf.split(train, train["interest_level"]):  # you have to specify targer variable as well.. X,X["target"]
    # Obtain training and testing folds
    cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]  
    print('Fold: {}'.format(fold))
    print('CV train shape: {}'.format(cv_train.shape))
    print('Medium interest listings in CV train: {}\n'.format(sum(cv_train.interest_level == 'medium')))
    fold += 1    

# Great! Now you see that both size and target distribution are the same among the folds. 
# The general rule is to prefer Stratified K-Fold over usual K-Fold in any classification problem. 
# Move to the next lesson to learn about other cross-validation strategies!
    

### Validation usage
    
# Pri pouziti K-fold cross validacie si musime davat pozor na pouzitie pri casovych radoch. Nesmieme v casovych radoch pouzit buduce data, resp. 
    #testovacie data musia byt vzdy ako posledne resp. az po train datach (co sa casu tyka) 

# Time K-fold cross-validation

# Create TimeSeriesSplit object
time_kfold = TimeSeriesSplit(n_splits = 3)

# Sort train data by date
train = train.sort_values("date")

# Iterate through each split
fold = 0
for train_index, test_index in time_kfold.split(train):
    cv_train, cv_test = train.iloc[train_index], train.iloc[test_index]
    
    print('Fold :', fold)
    print('Train date range: from {} to {}'.format(cv_train.date.min(), cv_train.date.max()))
    print('Test date range: from {} to {}\n'.format(cv_test.date.min(), cv_test.date.max()))
    fold += 1    

#Fold : 0
#Train date range: from 2017-12-01 to 2017-12-08
#Test date range: from 2017-12-08 to 2017-12-16

#Fold : 1
#Train date range: from 2017-12-01 to 2017-12-16
#Test date range: from 2017-12-16 to 2017-12-24

#Fold : 2
#Train date range: from 2017-12-01 to 2017-12-24
#Test date range: from 2017-12-24 to 2017-12-31    
    


## Overall validation score

# Now it's time to get the actual model performance using cross-validation! How does our store item demand prediction model perform?
# Your task is to take the Mean Squared Error (MSE) for each fold separately, and then combine these results into a single number.
# For simplicity, you're given get_fold_mse() function that for each cross-validation split fits a Random Forest model and returns a list of MSE scores by fold. get_fold_mse() accepts two arguments: train and TimeSeriesSplit object.    

import inspect  
print(inspect.getsource(get_fold_mse))   

def get_fold_mse(train, kf):
    mse_scores = []
    
    for train_index, test_index in kf.split(train):
        fold_train, fold_test = train.loc[train_index], train.loc[test_index]

        # Fit the data and make predictions
        # Create a Random Forest object
        rf = RandomForestRegressor(n_estimators=10, random_state=123)

        # Train a model
        rf.fit(X=fold_train[['store', 'item']], y=fold_train['sales'])

        # Get predictions for the test set
        pred = rf.predict(fold_test[['store', 'item']])
    
        fold_score = round(mean_squared_error(fold_test['sales'], pred), 5)
        mse_scores.append(fold_score)
        
    return mse_scores

from sklearn.model_selection import TimeSeriesSplit
import numpy as np

# Sort train data by date
train = train.sort_values('date')

# Initialize 3-fold time cross-validation
kf = TimeSeriesSplit(n_splits=3)

# Get MSE scores for each cross-validation split
mse_scores = get_fold_mse(train, kf)

print('Mean validation MSE: {:.5f}'.format(np.mean(mse_scores)))
print('MSE by fold: {}'.format(mse_scores))
print('Overall validation MSE: {:.5f}'.format(np.mean(mse_scores) + np.std(mse_scores)))

#<script.py> output:
#    Mean validation MSE: 955.49186

#<script.py> output:
#    Mean validation MSE: 955.49186
#    MSE by fold: [890.30336, 961.65797, 1014.51424]

#<script.py> output:
#    Mean validation MSE: 955.49186
#    MSE by fold: [890.30336, 961.65797, 1014.51424]
#    Overall validation MSE: 1006.38784

################
## CHAPTER 3  ##
################


# Modelig: Preprocess, create new features, improve models, apply tricks  --> LOCAL VALIDATION 

# Feature engineering from prior experience / EDA / domain knowledge

# Feature types: numerical, categorical, datetime, coordinates (spacial data), text,images

# Creating features:
data = pd.concat([train, test])

train = data[data.id.isin(train.id)]
test = data[data.id.isin(test.id)]

# Arithmetical features
price_per_bedroom = price / rooms_number

# Datetime features
dem["date"] = pd.to_datetime(dem["date"])  ### then apply .dt atribute to obtain any info (hour / minute / second / day / weekday /year /month ..) 

dem["date"].dt. / year,month,weekofyear, dayofyear, day, dayofweek   

# Poznamka dayofweek (Monday = 0, Tuesday = 1 ....)


## Arithmetical features

import inspect  
print(inspect.getsource(get_kfold_rmse))   

def get_kfold_rmse(train):
    mse_scores = []

    for train_index, test_index in kf.split(train):
        train = train.fillna(0)
        feats = [x for x in train.columns if x not in ['Id', 'SalePrice', 'RoofStyle', 'CentralAir']]
        
        fold_train, fold_test = train.loc[train_index], train.loc[test_index]

        # Fit the data and make predictions
        # Create a Random Forest object
        rf = RandomForestRegressor(n_estimators=10, min_samples_split=10, random_state=123)

        # Train a model
        rf.fit(X=fold_train[feats], y=fold_train['SalePrice'])

        # Get predictions for the test set
        pred = rf.predict(fold_test[feats])
    
        fold_score = mean_squared_error(fold_test['SalePrice'], pred)
        mse_scores.append(np.sqrt(fold_score))
        
    return round(np.mean(mse_scores) + np.std(mse_scores), 2)

# Look at the initial RMSE
print('RMSE before feature engineering:', get_kfold_rmse(train))

# Add total area of the house
train['TotalArea'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']
print('RMSE with total area:', get_kfold_rmse(train))

# Add garder area of the property
train['GardenArea'] = train['LotArea'] - train['1stFlrSF']
print('RMSE with garden area:', get_kfold_rmse(train))

# Add total number of bathrooms
train['TotalBath'] = train['FullBath'] + train['HalfBath']
print('RMSE with number of bathrooms:', get_kfold_rmse(train))

# Nice! You've created three new features. Here you see that house area improved the RMSE by almost $1,000. 
# Adding garden area improved the RMSE by another $600. 
# However, with the total number of bathrooms, the RMSE has increased. 
# It means that you keep the new area features, but do not add "TotalBath" as a new feature. 
# Let's now work with the datetime features!


## Date features
# Concatenate train and test together
taxi = pd.concat([train,test])

# Convert pickup date to datetime object
taxi['pickup_datetime'] = pd.to_datetime(taxi["pickup_datetime"])

# Create day of week feature
taxi['day_of_week'] = taxi['pickup_datetime'].dt.dayofweek

# Create hour feature
taxi['hour'] = taxi['pickup_datetime'].dt.hour

# Split back into train and test
new_train = taxi[taxi.id.isin(test.id)]
new_test = taxi[taxi.id.isin(train.id)]

## Categorical features

# Label encoding - A->0 , B->1, C -> 2

# One-hot encoding -  CAT==A 0/1 .. dummy premenne  pomocou ohe = pd.get_dummies(df["var"], prefix = "ohe_cat")
#  after encoding, drop column "var" from df
#  concat df with ohe :  pd.concat([df, ohe])

# Binary features = flag

# Other encoding approaches:
   #Backward difference encoding, M-estimate, BaseN, Onehot, Binary, Ordinal, CatBoost encoder, WOE, James-Stein encoder, Hashing, Sum coding,


### Encoding 
   
# Concatenate train and test together
houses = pd.concat([train,test])

# Label encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

# Create new features
houses['RoofStyle_enc'] = le.fit_transform(houses['RoofStyle'])
houses['CentralAir_enc'] = le.fit_transform(houses['CentralAir'])

# Look at new features
print(houses[['RoofStyle', 'RoofStyle_enc', 'CentralAir', 'CentralAir_enc']].head())

# All right! You can see that categorical variables have been label encoded. 
# However, as you already know, label encoder is not always a good choice for categorical variables. 
# Let's go further and apply One-Hot encoding.


### One-Hot encoding

#The problem with label encoding is that it implicitly assumes that there is a ranking dependency between the categories. 
#So, let's change the encoding method for the features "RoofStyle" and "CentralAir" to one-hot encoding. 
#Again, the train and test DataFrames from House Prices Kaggle competition are already available in your workspace.

#Recall that if you're dealing with binary features (categorical features with only two categories) it is suggested to apply label encoder only.
#Your goal is to determine which of the mentioned features is not binary, and to apply one-hot encoding only to this one.

# Concatenate train and test together
houses = pd.concat([train, test])

# Look at feature distributions
print(houses['RoofStyle'].value_counts())
print(houses['CentralAir'].value_counts())   # --> binary

# Concatenate train and test together
houses = pd.concat([train, test])

# Label encode binary 'CentralAir' feature
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
houses['CentralAir_enc'] = le.fit_transform(houses['CentralAir'])

# Create One-Hot encoded features
ohe = pd.get_dummies(houses['RoofStyle'], prefix='RoofStyle')

# Concatenate OHE features to houses
houses = pd.concat([houses, ohe], axis=1)

# Look at OHE features
print(houses[[col for col in houses.columns if 'RoofStyle' in col]].head(3))

# Remember to drop the initial string column, because models will not handle it automatically. 
# OK, we're done with simple categorical encoders. 
# Let's move to the target encoder!

#### Target encoding

# High cardinality categorical features -> at least 10 distinct categories 

# Label encoder provides distinct number for each category
# One hot encoder creates many new fetures
# Target encoding to the rescue

## 1) Mean targer encoding  -> most commonly used on Kaggle
#     #1) Calculate mean of the train, apply to the tes.  Pre kazdu kategoriu target rate 
#     #2) Split train to K folds, calculate mean on k-1 folds and applyt to the k-th fold
#     #3) Add mean target encoded feature to the model

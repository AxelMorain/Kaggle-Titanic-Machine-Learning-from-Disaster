# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 16:44:32 2020

Axel Morain

Titanic Project

File name: Data Exploration.py

File Order:
    Data Exploration <=
    Function to Test Different ANN and Save the Best One
    Testing Different Neural Networks
    Runing models on Kaggle's test dataset

•Status: Working and ready to upload

•What?:
    -Data cleaning, data exploration and running a few models,
    -I got some isparation from this tutorial:
        https://www.kaggle.com/saadmuhammad17/a-beginners-guide-to-data-science-top-3

•Why?:
        
•How?: 
    -Data Cleaning:
        Dealing with missing values:
            Statistics were used to estimate the missing Age
            Missing Fare were replaced by the median
            Missing Embarked were replaced by the mode
            The entire Cabin column was deleted
        Dealing wiht other variables:        
            Tiles were extracted out of the Name column and put into a new column
            Column Name was deleted
            The entire column Ticket was deleted
            
    -Data Preparation:
        One Hot Encoding was apply to the variables: Sex, Embarked and Title
        All variables were z-scored
        
    -Model Tested and Results:
        Decision Tree =>
        KNN with 3 nearest neighbors =>
        KNN with 5 nearest neighbors =>
        Random Forest =>
        Perceptron =>
        gaussian =>
    
    
•Results:
    -Two models were selected, Decision Tree and Random Forrest, both scoring
    96.41 and 89.56 respectively on the training set. Suprisingly, running the 
    models on the test set and submitting the results lead to a 70% and 74%
    accuracy for the Decision Tree and Random Forrest respectively. This lead
    to the conclusion that Dcision Tree does overfit.
    
    
***  TO DO / Notes  ***
- From https://www.kaggle.com/c/titanic/data download the data
- 
-
-

"""
import numpy as np
import pandas as pd
pd.set_option('display.expand_frame_repr', True)
#pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 100)

'''______________________________________________________________________________________________'''


'''
#
#  Imporint Data
#
'''

raw_train_data = pd.read_csv('train.csv') #Data set from https://www.kaggle.com/c/titanic
raw_test_data = pd.read_csv('test.csv')   #Data set from https://www.kaggle.com/c/titanic
raw_all_data = pd.concat([pd.DataFrame(raw_train_data.drop('Survived', axis = 1)),
                         raw_test_data])


# Quick data dxploration
print(raw_train_data.head(5))
print(raw_train_data.iloc[1:5,1:3])
print(raw_train_data.head())

summary_raw_all_data = raw_all_data.describe().round()
print(summary_raw_all_data)


'''
#
# Data Cleaning
#
'''

missing_val = pd.DataFrame({'Column_names' : raw_all_data.columns,
                            'Num of missing values' : raw_all_data.isnull().sum()})
print(missing_val)
# 3 columns have missing values: Age, Cabin and Embarked

# Fixing Age missing values
# We are going to use correlations between age and other variables to predict 
#the missing values of Age

""" ---- Make the Sex variable numerical --- """
all_data = raw_all_data.copy().reset_index(drop = True)

for i in range(all_data.shape[0]):
    if all_data.Sex[i] == 'male':
        all_data.Sex[i] = 1
    else:
        all_data.Sex[i] = 0

all_data.Sex = all_data.Sex.astype('int')


df_coor = all_data.corr().abs().unstack()['Age'].sort_values(ascending = False)

df_coor = pd.DataFrame ({'Feature 1': ['Age','Age','Age','Age','Age','Age','Age'],
                         'Feature 2': df_coor.index,
                         'Correlation Coeficient': df_coor.values })

# According to Class and SibSp we are estimating the Age

age_by_sex_and_SibSp = all_data.groupby(['Sex', 'SibSp']).median()['Age']
# For some reason there is no Age value when Sex == 0 and SibSp == 8
# Let put the women median age instead
woman_median_age = all_data.Age.loc[(all_data.Sex == 0)].median()
age_by_sex_and_SibSp[0][8] = woman_median_age
del woman_median_age

for i in all_data[all_data.Age.isnull()].index.values:
    if (all_data.Sex[i] == 0 ) & (all_data.SibSp[i] == 0):
        all_data.Age[i] = age_by_sex_and_SibSp[0][0]
    if (all_data.Sex[i] == 0 ) & (all_data.SibSp[i] == 1):
        all_data.Age[i] = age_by_sex_and_SibSp[0][1]    
    if (all_data.Sex[i] == 0 ) & (all_data.SibSp[i] == 2):
        all_data.Age[i] = age_by_sex_and_SibSp[0][2]    

    if (all_data.Sex[i] == 0 ) & (all_data.SibSp[i] == 3):
        all_data.Age[i] = age_by_sex_and_SibSp[0][3]    
    if (all_data.Sex[i] == 0 ) & (all_data.SibSp[i] == 4):
        all_data.Age[i] = age_by_sex_and_SibSp[0][4]     
    if (all_data.Sex[i] == 0 ) & (all_data.SibSp[i] == 5):
        all_data.Age[i] = age_by_sex_and_SibSp[0][5]     
    if (all_data.Sex[i] == 0 ) & (all_data.SibSp[i] == 8):
       all_data.Age[i] = age_by_sex_and_SibSp[0][8] 

    if (all_data.Sex[i] == 1 ) & (all_data.SibSp[i] == 0):
        all_data.Age[i] = age_by_sex_and_SibSp[1][0]
    if (all_data.Sex[i] == 1 ) & (all_data.SibSp[i] == 1):
        all_data.Age[i] = age_by_sex_and_SibSp[1][1]    
    if (all_data.Sex[i] == 1 ) & (all_data.SibSp[i] == 2):
        all_data.Age[i] = age_by_sex_and_SibSp[1][2]    

    if (all_data.Sex[i] == 1 ) & (all_data.SibSp[i] == 3):
        all_data.Age[i] = age_by_sex_and_SibSp[1][3]    
    if (all_data.Sex[i] == 1 ) & (all_data.SibSp[i] == 4):
        all_data.Age[i] = age_by_sex_and_SibSp[1][4]     
    if (all_data.Sex[i] == 1 ) & (all_data.SibSp[i] == 5):
        all_data.Age[i] = age_by_sex_and_SibSp[1][5]            
    if (all_data.Sex[i] == 1 ) & (all_data.SibSp[i] == 8):
       all_data.Age[i] = age_by_sex_and_SibSp[1][8]

missing_val_all_data = pd.DataFrame({'Column_names' : all_data.columns,
                            'Num of missing values' : all_data.isnull().sum()})
missing_val_all_data

# Lets deal with Fare, only one is missing
# Replace it with the median

all_data.Fare = all_data.Fare.fillna(all_data.Fare.median())


missing_val_all_data = pd.DataFrame({'Column_names' : all_data.columns,
                            'Num of missing values' : all_data.isnull().sum()})
missing_val_all_data

# For Embarked, replace the missing values by the most frequent value

all_data.Embarked = all_data.Embarked.fillna(all_data.Embarked.value_counts().index[0])

missing_val_all_data = pd.DataFrame({'Column_names' : all_data.columns,
                            'Num of missing values' : all_data.isnull().sum()})
missing_val_all_data

# Delete Cabin, there is not enough information to predict what so ever.
all_data = all_data.drop(['Cabin'], axis = 1)

# No mor mising values !
del age_by_sex_and_SibSp, df_coor, i, missing_val, missing_val_all_data, summary_raw_all_data


'''
#
# Data Preparation
#
'''

summary_all_data = all_data.describe(include = 'all')

# Dealing with Name
# According to summary_all_data all but one names are uniques
# but we can get the title out of it which might be usefull.

all_data['Title'] = all_data['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]

all_data.Title.value_counts()
percentage_4_first_title = round(all_data.Title.value_counts()[0:4].sum() / all_data.shape[0] * 100)
print('The 4 first title represent:',percentage_4_first_title, '%'   )

# The 4 first titles will be kept, all the other titles will be combined into a
#single new title called: Others

Condition1 = all_data.Title == 'Mr'
Condition2 = all_data.Title == 'Miss'
Condition3 = all_data.Title == 'Mrs'
Condition4 = all_data.Title == 'Master'

all_data.Title.where(Condition1 | Condition2 | Condition3 | Condition4 ,
                     other =  'Other',
                     inplace = True)

all_data.Title.value_counts() # This looks right

del Condition1, Condition2 ,Condition3, Condition4, percentage_4_first_title

# Names: the usefull data was extracted from the Name columns so we can now 
#delete it as well.
# Drop PassenderId as well

all_data = all_data.drop(['Name', 'PassengerId'], axis = 1)

#One last check
missing_val_all_data = pd.DataFrame({'Column_names' : all_data.columns,
                            'Num of missing values' : all_data.isnull().sum()})
missing_val_all_data

#Ticket
unique_ticket = set(all_data.Ticket)
print('The percentage of unique ticket is:',
      round(len(unique_ticket) / all_data.shape[0] * 100))
# let's delete this variable, I do not think valueable information can be obtain
del unique_ticket

all_data = all_data.drop(['Ticket'], axis = 1)

# Cleaning Time
del missing_val_all_data,  summary_all_data


'''
#
# Looking for correlations in the train_data
#
'''

train_data = all_data.head(raw_train_data.shape[0])
test_data = all_data.tail(raw_test_data.shape[0])

train_data = pd.concat([raw_train_data.Survived, train_data  ], axis = 1)

#Create a df with all correlations

col_name = np.array(['Pclass', 'Sex' ,'Age', 'SibSp', 'Parch','Fare'])
correlations = np.array([train_data.Survived.corr(train_data.iloc[:,1]).round(2),
                         train_data.Survived.corr(train_data.iloc[:,2]).round(2),
                        train_data.Survived.corr(train_data.iloc[:,3]).round(2),
                        train_data.Survived.corr(train_data.iloc[:,4]).round(2),
                        train_data.Survived.corr(train_data.iloc[:,5]).round(2),
                        train_data.Survived.corr(train_data.iloc[:,6]).round(2)])

df_corr = pd.DataFrame({'Attributes': col_name,
                        'Correlations':correlations})
df_corr.Correlations = df_corr.Correlations.abs()
df_corr = df_corr.sort_values(['Correlations'], ascending = False )

del correlations, col_name

'''
#
# Data preparation for models
#
'''

#One Hot Encoding for Sex, Embarked and Title
all_data = pd.get_dummies(all_data, columns=['Pclass'], prefix=['Pclass'])
all_data = pd.get_dummies(all_data, columns=['Sex'], prefix=['Sex'])
all_data = pd.get_dummies(all_data, columns=['SibSp'], prefix=['SibSp'])
all_data = pd.get_dummies(all_data, columns=['Parch'], prefix=['Parch'])
all_data = pd.get_dummies(all_data, columns=['Embarked'], prefix=['Embarked'])
all_data = pd.get_dummies(all_data, columns=['Title'], prefix=['Title'])
all_data.dtypes
all_data = all_data.astype('int')
all_data.dtypes

#Calculating the z-score
all_data = (all_data - all_data.mean())/all_data.std()

# Splitting the data and adding the survived variable to the training data
train_data = all_data.head(raw_train_data.shape[0])
test_data = all_data.tail(raw_test_data.shape[0])
train_data = pd.concat([raw_train_data.Survived, train_data  ], axis = 1)
train_data.dtypes

x_train = train_data.iloc[:, 1:].values
y_train = train_data.iloc[:, 0].values

x_test = test_data

'''
#
# Export the new test and train sets
#
'''

train_data.to_csv('clean_train_data.csv')

x_train = pd.DataFrame(x_train)
y_train = pd.DataFrame(y_train)
x_test = pd.DataFrame(x_test)

x_train.to_csv('x_train_data.csv', index = False)
y_train.to_csv('y_train_data.csv', index = False)
x_test.to_csv('x_test_data.csv', index = False)

'''
#
# Model Testing
#
'''


#Models for checking, might not use all

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# Decision Tree
decision_tree = DecisionTreeClassifier(criterion='entropy', splitter='random') 
decision_tree.fit(x_train, y_train)  
dt_pred = decision_tree.predict(x_test)  
dt_score = round(decision_tree.score(x_train, y_train) * 100, 2)
print("score:", dt_score, "%")

# KNN
knn = KNeighborsClassifier(n_neighbors = 5) 
knn.fit(x_train, y_train)  
knn_pred = knn.predict(x_test)  
knn_score1 = round(knn.score(x_train, y_train) * 100, 2)
print("score:", knn_score1, "%")

# KNN
knn = KNeighborsClassifier(n_neighbors = 3) 
knn.fit(x_train, y_train)  
knn_pred = knn.predict(x_test)  
knn_score = round(knn.score(x_train, y_train) * 100, 2)
print("score:", knn_score, "%")

# Random Forest
random_forest = RandomForestClassifier(criterion = "gini", 
                                       min_samples_leaf = 1, 
                                       min_samples_split = 10,   
                                       n_estimators=100, 
                                       max_features='auto', 
                                       oob_score=True, 
                                       random_state=1, 
                                       n_jobs=-1)

random_forest.fit(x_train, y_train)
rf_pred = random_forest.predict(x_test)

rf_score = round(random_forest.score(x_train, y_train) * 100, 2)
print("score:", rf_score, "%")

# Perceptron
perceptron = Perceptron(max_iter=5)
perceptron.fit(x_train, y_train)

percep_pred = perceptron.predict(x_test)

percep_score = round(perceptron.score(x_train, y_train) * 100, 2)
print("score:", percep_score, "%")

#gaussian
gaussian = GaussianNB() 
gaussian.fit(x_train, y_train)  
nb_pred = gaussian.predict(x_test)  
nb_score = round(gaussian.score(x_train, y_train) * 100, 2)
print("score:", nb_score, "%")

'''
#
# x_test Predictions
#
'''

# Decision Tree is doing the best, let's apply the model
test_results_decision_tree = pd.DataFrame(decision_tree.predict(x_test))
index_test_data = pd.DataFrame(raw_test_data.PassengerId.values)

# Formating for submition
Final_Prediction2 = pd.concat( [ index_test_data, test_results_decision_tree], axis = 1)
Final_Prediction2.columns = ['PassengerId','Survived']

# Exporting
Final_Prediction2.to_csv('Final_Prediction.csv', index = False)

# Try Random Forest
test_results_random_forest = pd.DataFrame(random_forest.predict(x_test))
Final_Prediction_random_forest = pd.concat( [ index_test_data, test_results_random_forest],
                                           axis = 1)
Final_Prediction_random_forest.columns = ['PassengerId','Survived']
# Exporting
Final_Prediction_random_forest.to_csv('Final_Prediction.csv', index = False)















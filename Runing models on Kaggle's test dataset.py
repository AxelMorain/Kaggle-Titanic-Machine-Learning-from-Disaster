# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 22:03:41 2020

Axel Morain

Titanic Project

•File name: Runing models on Kaggle's test dataset.py

•File Order:
    Data Exploration 
    Function to Test Different ANN and Save the Best One 
    Testing Different Neural Networks 
    Runing models on Kaggle's test dataset <=

•Status:  Working and ready to upload

•What?:
    Runnig the best model from Allconfiguration2 with data from Data Exploration2
    on the test data of the project and outputting the results to be uploaded 
    to Kaggle

•Why?: To get results and upload them
    
•How?: Just watch
    
•Results:  
    
    
***  TO DO / Notes  ***
- For unknown reasons the model is not compatible with the data set
but Configurations_test_Best_Model_with_Weights_Resets works...
- Check how the data sets on which the models are trained
-
-

"""
# Those are all the libraries I used for this project:
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
pd.set_option('display.expand_frame_repr', True)
#pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 100)
import time

'''__________________________________________________________________________________________'''


''' Import Model and Data '''

model = keras.models.load_model('Configurations_test' + '_Best_Model')

x_test = pd.read_csv('x_test_data.csv')

''' Apply '''

results = model.predict(x_test)
results = pd.DataFrame(results)
results = results.round() # Some rounding leads to -1 which should be 0

for i in range(0, results.shape[0]):
    if results[0][i] == -1:
        results[0][i] = 0

''' Formating for exportation '''

# Get the indeces
test_data = pd.read_csv('test.csv', index_col=False)

results = pd.concat([test_data.PassengerId, results], axis = 1)
results.columns = ['PassengerId', 'Survived']
results.Survived = results.Survived.astype('int')


results.to_csv('Final_Predictions.csv', index=False) # Ready to be uploaded to Kaggle
















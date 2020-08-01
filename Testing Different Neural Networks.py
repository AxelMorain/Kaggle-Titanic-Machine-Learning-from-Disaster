# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 13:49:08 2020

Axel Morain

Titanic Project

•File name: Testing Different Neural Networks.py

•File Order:
    Data Exploration 
    Function to Test Different ANN and Save the Best One 
    Testing Different Neural Networks <=
    Runing models on Kaggle's test dataset

•Status: Working and ready to upload

•What?:
     - Run and time the function created in 'Function to Test Different ANN and Save the Best One.py'

•Why?: 
    Trying different model architecture and parameters shuch as activation functions
    and optimizer is one way to find the best model
        
•How?:

    
•Results:

    
***  TO DO / Notes  ***
- 
- 
-
-
"""

import time

'''______________________________________________________________________________________________'''


import Function_to_Test_Different_ANN_and_Save_the_Best_One as MyFunctions

# Time it just for fun
start_time = time.time()

MyFunctions.Test_All_ANN_in_Configuratoins_with_Best_Model_Save('Configurations_test')# don't include the'.csv'

end_time = time.time()

print('Total time =',end_time - start_time)







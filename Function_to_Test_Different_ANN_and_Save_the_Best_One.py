# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 17:53:17 2020

Axel Morain

Titanic Project

•File name:     Function_to_Test_Different_ANN_and_Save_the_Best_One.py

•File Order:
    Data Exploration 
    Function to Test Different ANN and Save the Best One <=
    Testing Different Neural Networks
    Runing models on Kaggle's test dataset

•Status: Working and ready to upload

•What?:
    -Creates a funciton that runs, and stores results of different ANN artichitecture and setting,
    The neural networks were arbitrary limited to 2 hidden layers.
    -The input df countains information about architecture shape and setting that wants to be tried
    -There are two ouputs:
        The modified input file wiht test results as well as number of epoch
        The model which lead to the highest accuracy is saved as a folder

•Why?: 
    Trying different model architecture and parameters shuch as activation functions
    and optimizer is one way to find the best model
        
•How?:
    -The ANN imports its parameters from a file containing the following informations: 
    Hidden_Layer_1,Hidden_Layer_2, Activation_Func_1,Activation_Func_2,
    Activation_Func_Output_Layer,Loss_function,Optimizer,Metrix,Repeat. 
    -All parameters are self explainatory except for Repeat. Repeat represent
    the number of time the same model is giong to be ran. This is an atempt to 
    quantify the stochastic behavior of ANN.
    -This arose a question, what if the ANN is retrained without resetting its
    weights? Will the accuracy increase? Will it lead to overfitting? I have
    answered those questions myself in a different file that I might upload
    later. You are more than welcome to try it out by yourself =)
    - A test Configuration file should be in the Github as well. It countains
    a few different models, you are more than welcome to elaborate on it.
    
•Results:

    
***  TO DO / Notes  ***
- 
- 
-
-

"""
'''______________________________________________________________________________________________'''

''' Creating a Funciton With Auto-Save'''

def Test_All_ANN_in_Configuratoins_with_Best_Model_Save(Configurations_file):
    #Enter the file name without the '.csv' extention
    
    ''' This function also can reinitiase weights after each run, but it is turned off
    To turn it on, delete the '#' on the lines 124 and 187 to enable the code to be read '''
    
    import numpy as np
    import pandas as pd
    from keras.models import Sequential
    from keras.layers import Dense
    import tensorflow as tf
    from sklearn.preprocessing import StandardScaler
    pd.set_option('display.expand_frame_repr', True)
    #pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.width', 100)

    '''______________________________________________________________________________________________'''
    
    
    ''' Import training data '''
    data = pd.read_csv('clean_train_data.csv', index_col = False)
    data = data.iloc[:, 1:]
    
    ''' Preparing the Data '''
    #Splitting the data
    data = data.sample(frac=1)
    data = data.reset_index(drop = True)
    
    y = data.iloc[:,0]
    y = pd.DataFrame(y)
    X = StandardScaler().fit_transform(data.drop('Survived', axis = 1))
    X = pd.DataFrame(X)
    y_train = y.head(round(len(data)*(80/100)))
    X_train = X.head(round(len(data)*(80/100)))
    
    y_test = y.tail(round(len(data)*(20/100)))
    X_test = X.tail(round(len(data)*(20/100)))
    
    y_test = y_test.reset_index(drop = True)
    X_test = X_test.reset_index(drop = True)
    

    
    ''' Import Configurations '''
    global Configurations
    
    Configurations = pd.read_csv(Configurations_file + '.csv', index_col=False)
   
    for i in range(0, Configurations.shape[0]):
        print ('i = ',i)
        
        ''' Building the Neural Networks '''
        model = Sequential()
        model.add(Dense(int(Configurations.Hidden_Layer_1[i]), input_dim= X.shape[1],
                        activation = Configurations.Activation_Func_1[i]))
        model.add(Dense(int(Configurations.Hidden_Layer_2[i]),
                        activation = Configurations.Activation_Func_2[i]))
        model.add(Dense(1, activation= Configurations.Activation_Func_Output_Layer[i]))
        
        model.compile(loss=Configurations.Loss_function[i],
                      optimizer=Configurations.Optimizer[i],
                      metrics=[Configurations.Metrix[i]])
        
        #Save the original randomised weights to reset them later if needed/wanted
#        model.save_weights('model_random_original_weights.h5')
        
        ''' Early Stop '''
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

        j = 0
        # To average the results of exactly similar models repeated int(Configurations.Repeat[i]) times
        #they are stored in an array
        #The numbers are place holders, it could all be NA's
        global Score_of_Similar_model
        Score_of_Similar_model = np.array(list(range(0,int(Configurations.Repeat[i]))))

        Number_of_epoch = np.array(list(range(0,int(Configurations.Repeat[i]))))
        # The number of epoch between the int(Configurations.Repeat[i]) similar models
        #are not averaged, only the last one is saved, they were not a lot of variance
        #between them and it is not an extrimely important value...

        ''' Run the same model j number of time '''
        while j < int(Configurations.Repeat[i]):  
            print('\t','j =: ',j)
            
            ''' Trainning the model '''
            History = model.fit(X_train, y_train, epochs = 1000, callbacks = [callback], verbose = 0)
            
            ''' Checking the Model's Performance '''            
            y_pred = model.predict(X_test)            
            y_pred = pd.DataFrame(y_pred)
            y_pred = round(y_pred)
            
            accuracy = np.array(list(range(0,y_pred.shape[0])))
            
            '''' Comparing the actual model accuracy to the previous best model accuracy '''
            # Getting the accuracy of the freshly ran model
            for k in range(0,y_pred.shape[0]):
                if y_pred.iloc[k,0] == y_test.iloc[k,0]:
                    accuracy[k] = 1
                else:
                    accuracy[k] = 0            
            accuracy = accuracy.sum() / accuracy.shape[0] *100
            
            #Create a value for previous_best_model_accuracy 
            try:
                previous_best_model_accuracy = np.load('previous_best_model_accuracy.npy')
            except:
                previous_best_model_accuracy = float(0)
            
            #Compare actual model accuracy to previous model accuracy
            name_of_the_best_model = Configurations_file + '_Best_Model'
            
            if accuracy > previous_best_model_accuracy:
                ''' Saving and ouputting the best model '''
                model.save(name_of_the_best_model)
                print(accuracy)
                previous_best_model_accuracy = accuracy
                np.save('previous_best_model_accuracy.npy', previous_best_model_accuracy )

                        
            # Put each results in a vector
            Score_of_Similar_model[j] = accuracy
            Number_of_epoch[j] = len(History.history['loss'])
            
            j = j + 1
            # Reset the weights to the original randomised weights if needed/wanted
            #model.load_weights('model_random_original_weights.h5')
        
        Score_of_Similar_model = pd.DataFrame(Score_of_Similar_model)
        Score_of_Similar_model = Score_of_Similar_model.describe()
        #Puts the results in the df Configuratoins
        Configurations.iloc[i,9] = Score_of_Similar_model.iloc[1,0]     #Mean
        Configurations.iloc[i,10] = Score_of_Similar_model.iloc[2,0]    #std
        Configurations.iloc[i,11]  = Score_of_Similar_model.iloc[3,0]   #min
        Configurations.iloc[i,12]  = Score_of_Similar_model.iloc[4,0]   #Q1
        Configurations.iloc[i,13]  = Score_of_Similar_model.iloc[5,0]   #Q2
        Configurations.iloc[i,14]  = Score_of_Similar_model.iloc[6,0]   #Q3
        Configurations.iloc[i,15]  = Score_of_Similar_model.iloc[7,0]   #Max
        Configurations.iloc[i,16]  = len(History.history['loss'])       # Number of epoch
                                                                        #of the last model
        
        
    Configurations.to_csv(Configurations_file + '_and_Results.csv')
    previous_best_model_accuracy = 0
    np.save('previous_best_model_accuracy.npy', previous_best_model_accuracy )
    

    


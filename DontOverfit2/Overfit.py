# Starter code for Kaggle - Don't Overfit! II dataset.
# 
# Objective: make predictions on a dataset after only having trained a model on ~10% of it. Don't overfit.
# 
# By Blake French


import os
import re
import scipy as sp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


# Display the data.
def display_data(dataframe):
    
    # print(dataframe.info())
    print(dataframe.describe())
    # print(dataframe.corr())
    
    # Test all column data for normality using Shapiro-Wilk and K^2 tests.
    alpha = 0.05
    SW_results = []
    K2_results = []
    normality_pass = []
    for col in dataframe.columns:
        stat, p = sp.stats.shapiro(dataframe[col])
        SW_results.append(p)
        temp = 0
        if p>=alpha:
            temp = 1
        stat, p = sp.stats.normaltest(dataframe[col])
        K2_results.append(p)
        if p>=alpha:
            normality_pass.append(temp * 1)
        else:
            normality_pass.append(0)
            
    
    # Plot SW test p-values.
    temp_df = pd.DataFrame(SW_results, index=range(len(dataframe.columns)), columns=['SW_results'])
    temp_df.hist(color='green', bins=len(dataframe.columns), figsize=(8, 4))
    plt.show()
    # Plot K^2 test p-values.
    temp_df = pd.DataFrame(K2_results, index=range(len(dataframe.columns)), columns=['K2_results'])
    temp_df.hist(color='blue', bins=len(dataframe.columns), figsize=(8, 4))
    plt.show()
    # Plot pass/fail of both tests for each 
    temp_df = pd.DataFrame(normality_pass, index=range(len(dataframe.columns)), columns=['NormalityPassFail'])
    temp_df.hist(color='red', bins=2, figsize=(8, 4))
    plt.show()


# Clean the data.
def clean_data(dataframe):
    
    # Remove 'target' if present.
    target_present = False
    if 'target' in dataframe:
        target_present = True
        target = dataframe.pop('target')
    
    # Perform any cleaning.
    pass
    
    # Add polynomial features.
    '''id = dataframe.pop('id')
    poly = PolynomialFeatures(2)
    temp = poly.fit_transform(dataframe)
    poly_header = poly.get_feature_names(dataframe.columns)
    dataframe = pd.DataFrame(data=temp, index=dataframe.index, columns=poly_header)
    dataframe = pd.concat([id, dataframe], axis=1)'''
    
    # Perform feature scaling.
    scaler = StandardScaler()
    id = dataframe.pop('id')
    dataframe[dataframe.columns] = scaler.fit_transform(dataframe[dataframe.columns])
    dataframe = pd.concat([id, dataframe], axis=1)
    
    # Merge target if present in original data.
    if target_present:
        dataframe = pd.concat([target, dataframe], axis=1)
    
    return dataframe


# Main execution thread.
if __name__=='__main__':
    
    # Read all data.
    top_folder = '.'
    df = pd.read_csv(os.path.join('.', 'train.csv'))
    training_output = df.pop('target') # Remove output variable
    training_ids = list(df['id']) # Get ids for training set.
    df_final = pd.read_csv(os.path.join('.', 'test.csv'))
    df_all = pd.concat([df, df_final], axis=0, sort=False)
    
    # Display data.
    # display_data(df_all)
    
    # Clean data.
    df_all = clean_data(df_all)
    df = df_all[df_all['id'].isin(training_ids)]
    df_ids = df.pop('id')
    df_final = df_all[~df_all['id'].isin(training_ids)]
    df_final_ids = df_final.pop('id')
    
    # Fit model.
    params = {
        #'gamma':[1e-6],
        'kernel':['rbf', 'linear'],
        # 'C':[1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3]
    }
    grid = GridSearchCV(SVC(gamma='scale', C=1e-2), params, cv=10)
    grid.fit(df, training_output)
    print('Best SVM parameter values:')
    print(grid.best_params_)
    print('Best prediction score: ' + str(round(grid.best_score_, 3)))
    print()
    predictions = grid.predict(df_final)
    
    # Save predictions to output file.
    temp = pd.DataFrame(predictions, columns=['target'])
    temp.insert(0, 'id', df_final_ids)
    temp['target'] = temp['target'].astype('int')
    temp.to_csv('prediction.csv', header=list(temp), index=False)
    print('\nData saved.\n')

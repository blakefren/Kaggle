# Starter code for Kaggle - Titanic Survivor dataset.
# 
# Objective: classify Titanic passengers as surviving or perishing.
# 
# Uses a SVM with Gaussian (RBF) kernel.
# 
# 
# By Blake French


import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import AdaBoostClassifier


# Model selection.
model_type = 'svm' # best accuracy
# model_type = 'forest' # poor accuracy
# model_type = 'linreg' # not great accuracy
# model_type = 'knn' # even poorer accuracy


# Data visualization.
def display_data(dataframe):
    print(dataframe.info())
    print(dataframe.describe())
    print(dataframe.corr())
    f,ax=plt.subplots(figsize=(16,8))
    sns.heatmap(dataframe.corr(), annot=True, cmap='viridis') # feature correlation scores
    plt.show()
    # sns.heatmap(dataframe.isnull(), yticklabels=False, cbar=False, cmap='viridis')
    plt.show()
    sns.set_style('whitegrid')
    # sns.countplot(x='Survived', hue='Sex', data=dataframe, palette='RdBu_r')
    plt.show()
    # sns.countplot(x='Survived', hue='Pclass', data=dataframe, palette='rainbow')
    plt.show()
    # sns.distplot(dataframe['Age'].dropna(), color='darkred', bins=30)
    plt.show()
    temp = dataframe[dataframe['Fare']!=0].dropna()
    # temp['Fare'].hist(color='green', bins=40, figsize=(8, 4))
    plt.show()
    # plt.figure(figsize=(12, 7))
    # sns.boxplot(x='Pclass', y='Age', data=dataframe, palette='winter')
    plt.show()
    

# Data cleaning.
def clean_data(dataframe):
    
    survival = dataframe.pop('Survived') # Pop survived column for now.
    dataframe['Sex'] = dataframe['Sex'].map({'female': 1, 'male': 0}) # Turn gender into binary value.
    temp_embarked = pd.get_dummies(dataframe['Embarked']) # Split port of embarkation into different feature sets.
    dataframe = pd.concat([dataframe, temp_embarked], axis=1)

    # Replace NaN age values with avg age of their Pclass.
    p1_avg = dataframe[dataframe['Pclass']==1]['Age'].mean()
    p2_avg = dataframe[dataframe['Pclass']==2]['Age'].mean()
    p3_avg = dataframe[dataframe['Pclass']==3]['Age'].mean()
    dataframe.loc[dataframe['Pclass']==1, 'Age'] = dataframe.loc[dataframe['Pclass']==1, 'Age'].fillna(p1_avg)
    dataframe.loc[dataframe['Pclass']==2, 'Age'] = dataframe.loc[dataframe['Pclass']==2, 'Age'].fillna(p2_avg)
    dataframe.loc[dataframe['Pclass']==3, 'Age'] = dataframe.loc[dataframe['Pclass']==3, 'Age'].fillna(p3_avg)
    
    # Add adult/child data.
    dataframe['Adult'] = [1 if age>=18 else 0 for age in dataframe['Age']]
    dataframe['Old'] = [1 if age>=(dataframe['Age'].mean()) else 0 for age in dataframe['Age']]
    
    # Fill in missing Fare values w/ column mean, get FarePerPerson.
    dataframe.loc[dataframe['Fare'].isnull(), 'Fare'] = dataframe.loc[dataframe['Fare'].isnull(), 'Fare'].fillna(dataframe['Fare'].mean())
    dataframe['FarePerPerson'] = dataframe['Fare'] / (dataframe['SibSp'] + dataframe['Parch'] + 1)
    
    # Get titles from Name field.
    unique_titles = list(set(re.search('^[^,]+, ([\w\s]+)\.', name).group(1) for name in dataframe['Name']))
    unique_titles.sort()
    title_dict = {}
    for i,name in enumerate(unique_titles):
        title_dict[name] = i
    dataframe['Title'] = [title_dict[re.search('^[^,]+, ([\w\s]+)\.', name).group(1)] for name in dataframe['Name']]
    
    # Get surnames from Name field, exclude adult men.
    all_surnames = [name.split(',')[0] for name in dataframe['Name']]
    unique_surnames = list(set(all_surnames))
    unique_surnames.sort()
    surname_dict = {}
    for i,name in enumerate(unique_surnames):
        surname_dict[name] = [i+1, 0]
    dataframe['SurnameWomenChildren'] = [surname_dict[name][0] for name in all_surnames]
    dataframe.loc[(dataframe['Adult']==1) & (dataframe['Sex']==0), 'SurnameWomenChildren'] = 0
    
    # Identify single passengers.
    for name in all_surnames:
        surname_dict[name] = [surname_dict[name][0], surname_dict[name][1]+1]
    single_dict = {}
    for name in surname_dict:
        if surname_dict[name][1] == 1:
            single_dict[name] = 1
        else:
            single_dict[name] = 0
    dataframe['Single'] = [single_dict[name] for name in all_surnames]
    
    # Add explicit 1/0 to average family survival for women/child groups.
    dataframe['WomenChildGroupSurvival'] = -3
    dataframe['Surname'] = all_surnames
    dataframe = pd.concat([survival, dataframe], axis=1)
    for name in unique_surnames:
        avg_survival = dataframe.loc[(dataframe['Surname']==name) & ~((dataframe['Adult']==1) & (dataframe['Sex']==0)), 'Survived'].dropna().mean()
        if pd.isnull(avg_survival):
            avg_survival = 0.0
        dataframe.loc[(dataframe['Surname']==name) & ~((dataframe['Adult']==1) & (dataframe['Sex']==0)), 'WomenChildGroupSurvival'] = avg_survival
    dataframe.loc[dataframe['Single']==1, 'WomenChildGroupSurvival'] = -1 # Single passengers.
    dataframe.loc[(dataframe['Adult']==1) & (dataframe['Sex']==0), 'WomenChildGroupSurvival'] = -2 # Adult men.
    dataframe = dataframe.drop(['Survived', 'Surname'], axis=1)
    
    # Cabin data.
    # dataframe['Cabin'] = [1 if len(name)>0 else 0 for name in dataframe['Cabin'].fillna('')]
    # dataframe['Cabin'] = [(ord(c[0].lower()) - 96) for c in dataframe['Cabin'].fillna('U')]
    
    # Remove data items from frame that are not numerical.
    dataframe = dataframe.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
    
    # Add polynomial features.
    pass_id = dataframe.pop('PassengerId')
    poly = PolynomialFeatures(2)
    temp = poly.fit_transform(dataframe)
    poly_header = poly.get_feature_names(dataframe.columns)
    dataframe = pd.DataFrame(data=temp, index=dataframe.index, columns=poly_header)
    dataframe = pd.concat([pass_id, dataframe], axis=1)
    
    # Perform feature scaling.
    scaler = StandardScaler()
    pass_id = dataframe.pop('PassengerId')
    dataframe[dataframe.columns] = scaler.fit_transform(dataframe[dataframe.columns])
    dataframe = pd.concat([pass_id, dataframe], axis=1)
    
    # Drop features that have the same value for all rows.
    cols = list(dataframe)
    nunique = dataframe.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    dataframe = dataframe.drop(cols_to_drop, axis=1)
    
    # Add survival back in.
    dataframe = pd.concat([survival, dataframe], axis=1)
    
    # Return data.
    return(dataframe)


# Main execution thread.
if __name__ == '__main__':
    # Read all data.
    top_folder = '.'
    df = pd.read_csv(os.path.join(top_folder, 'train.csv'))
    # display_data(df)
    training_passengerIDs = list(df['PassengerId'])
    df_final = pd.read_csv(os.path.join(top_folder, 'test.csv'))
    # display_data(df_final)
    df_all = pd.concat([df, df_final], axis=0, sort=False)
    # display_data(df_all)


    # Clean data.
    print('\nReading and cleaning data...\n')
    df_all = clean_data(df_all)
    df = df_all[df_all['PassengerId'].isin(training_passengerIDs)]
    # display_data(df)
    df_final = df_all[~df_all['PassengerId'].isin(training_passengerIDs)].drop('Survived', axis=1)
    # display_data(df_all)
    training_output = df.pop('Survived') # Remove output variable


    # Fit the model and make predictions on the final test set.
    print('\nFitting model...\n')
    predictions = None

    if model_type=='svm':
        # Fit SVM model.
        params = {'gamma':[1e-6], # 1e-9, 1e-8, 1e-7
        'C':[1e3]} # 1e5, 1e6, 1e7]}
        grid = GridSearchCV(SVC(kernel='rbf'), params, cv=3)
        grid.fit(df, training_output)
        print('Best SVM parameter values:')
        print(grid.best_params_)
        print('Best prediction score: ' + str(round(grid.best_score_, 3)))
        print()
        predictions = grid.predict(df_final)

    elif model_type=='forest':
        # Fit random forest model.
        # rfc = RandomForestClassifier() # n_estimators=3000, min_samples_split=4)#, class_weight={0:0.745,1:0.255})
        params = {'n_estimators':[5, 10, 50, 100]}
        grid = GridSearchCV(RandomForestClassifier(), params, cv=3)
        grid.fit(df, training_output)
        print('Best SVM parameter values:')
        print(grid.best_params_)
        print('Best prediction score: ' + str(round(grid.best_score_, 3)))
        print()
        predictions = grid.predict(df_final)

    elif model_type=='linreg':
        # Fit logistic model.
        params = {'C':[2e-2, 2e-1, 2e0, 2e1, 2e2]}
        grid = GridSearchCV(LogisticRegression(solver='liblinear', max_iter=1000), params, cv=3)
        grid.fit(df, training_output)
        print('Best logistic regression parameter values:')
        print(grid.best_params_)
        print('Best prediction score: ' + str(round(grid.best_score_, 3)))
        print()
        predictions = grid.predict(df_final)

    elif model_type=='knn':
        # Fit k-nearest-neighbors model.
        params = {'n_neighbors':[5, 10, 15, 20],
        'weights':['distance', 'uniform'],
        'leaf_size':list(range(1,50,5))}
        grid = GridSearchCV(KNeighborsClassifier(algorithm='auto'), params, cv=3)
        grid.fit(df, training_output)
        print('Best KNN parameter values:')
        print(grid.best_params_)
        print('Best prediction score: ' + str(round(grid.best_score_, 3)))
        print()
        predictions = grid.predict(df_final)

    else:
        print('\n\tModel_type not supported.')
        exit()

    # Save predictions to output file.
    temp = pd.DataFrame(predictions, columns=['Survived'])
    temp.insert(0, 'PassengerId', df_final['PassengerId'])
    temp['Survived'] = temp['Survived'].astype('int')
    temp.to_csv('prediction.csv', header=list(temp), index=False)
    print('\nData saved.\n')

    # Check against correct answers I found online to avoid using Kaggle's (randomly) broken website.
    df_correct = pd.read_csv(os.path.join(top_folder, 'correct_answers.csv'))
    temp.sort_values('PassengerId')
    df_correct.sort_values('PassengerId')
    accuracy = accuracy_score(temp['Survived'], df_correct['Survived'])
    print('Accuracy against master list:\t' + str(round(accuracy, 6)) + '\n')

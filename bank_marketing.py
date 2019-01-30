# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 10:52:33 2019

@author: Varsha Choudhary
title: "Bank marketing campaign using SVM"
date: "January 18, 2019"
"""

#importing libraries for EDA and classification
import pandas as pd
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split

#import bank marketing csv file and bring it to right format. Importing headers as normal entry
df = pd.read_csv('C:/Users/Varsha/Desktop/JOB_STUDY_Material/Python/bank-full.csv', header = None)


# Delimiter is ";". But the program isn't aking it. Hence, reading the whole file splitting the data by ";" programitically 
df = df[0].astype(str)

#Splitting data by ";"
df = df.str.split(";", expand=True)

df.head(1)

#Saving headers
headers = df.iloc[0]

#Assigning headers
new_df  = pd.DataFrame(df.values[1:], columns=headers)

new_df.columns.values

#checking the data structure and datatypes of each column
new_df.shape
print(new_df.head(2))
new_df.dtypes

#make a list of all the columns
a= new_df.columns.tolist()

#drop the numeric columns for which no need to check for unique values
a.remove('age')
a.remove('balance')
a.remove('duration')
print(a)

#check column unique values of other variables
for col in a:
    new_df[col] = new_df[col].astype('category')
    print(sorted(new_df[col].unique()))
    
#Data Cleaning: Convert the columns  to right format        
print(sorted(new_df['job'].unique()))
new_df.columns = new_df.columns.str.strip()

#As the data had " at start and end of every text. Hence, removing it from entire datframe
new_df=new_df.replace('"','', regex=True)

#Changing columns "age", "balance", "duration" from data type "object" to "numeric" 
new_df[["age", "balance", "duration"]] = new_df[["age", "balance", "duration"]].apply(pd.to_numeric)
#ew_df["duration"] = new_df["duration"].apply(pd.to_numeric)


#Getting the 3 statistical moments for "age" and then for entire numeric columns.
new_df['age'].describe()
new_df.describe()

#Create dummy variables for categorial variables
#SVM can only operate on numeric values
dummies = pd.get_dummies(new_df[['marital', 'education', 'default','housing',\
                                 'loan','campaign','previous','poutcome','y']])
    
#Adding other numneric columns with dummy variables to create new dataframe 
d = pd.concat([new_df[['age','balance']], dummies], axis=1)

#Define X and Y for SVM
X = np.array(d.drop(['y_no', 'y_yes'], 1))
y = np.array(d[['y_no', 'y_yes']]) 

#Split train test data into 80:20 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) 
X_train.shape, y_train.shape
X_test.shape, y_test.shape

#Scale train data for better accuracy
X_scaled = preprocessing.scale(X_train)

#Train data on SVM
lin_clf = svm.LinearSVC()
lin_clf.fit(X_train, y_train)
clf = svm.SVC()
clf.fit(X_train, y_train)

#Check accuracy
accuracy = clf.score(X_test, y_test)
print(accuracy) 
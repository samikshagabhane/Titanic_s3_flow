#!/usr/bin/env python


import pandas as pd
import numpy as np

try:
    train = pd.read_csv('./train_mod.csv')
except Exception as e:
    print(e)


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age


train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)


train['Embarked'] = train['Embarked'].fillna('S')


train.drop('Cabin',axis=1,inplace=True)


train.dropna(inplace=True)


sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)


train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


train = pd.concat([train,sex,embark],axis=1)


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(train.drop(['Survived'],axis=1), 
                                                    train['Survived'], test_size=0.10, 
                                                    random_state=101)
print("split done")



from sklearn.ensemble import RandomForestClassifier


rf= RandomForestClassifier(n_estimators=500)
rf.fit(X_train,y_train)

print("creating pkl file")

import pickle as pkl
import os
print(os.getcwd())
pkl.dump(rf,open('./model_1.pkl','wb'))



rf_pre=rf.predict(X_test)

print(rf_pre)


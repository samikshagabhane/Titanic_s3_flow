#!/usr/bin/env python
import pandas as pd
import numpy as np
from compress_pickle import dump
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import os


try:
    train = pd.read_csv('./train_mod.csv')
except Exception as e:
    print(e)

train_df, target = train.drop(['Survived'],axis=1), train['Survived']
print("split done")

features_categoricas = ['Embarked', 'Sex', 'Pclass']
features_para_remover = ['Name', 'Cabin', 'Ticket', 'PassengerId']

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder())
])
preprocessor = ColumnTransformer(
    transformers=[
        ('Features categoricas', categorical_transformer, features_categoricas),
        ('Feature para remover', 'drop', features_para_remover)
])


model = Pipeline([('preprocessor', preprocessor),
                 ('clf', RandomForestClassifier(n_estimators=500)),
                 ])

model.fit(train_df, target)


print("creating pkl file")

dump(model,'./model_1.pkl',compression="lzma",set_default_extension=False)

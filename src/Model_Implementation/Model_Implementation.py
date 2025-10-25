#Implements a model to detect former farm land (maybe) using sklearn's RandomForestClassifier

import sklearn
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd

#Read in training data
training_data = pd.read_csv("data\\Training_Data.csv")

#Prepare feature matrix and labels
feature_columns = [col for col in training_data.columns if col not in ['ID', 'Northing', 'Easting', 'LTPC']]
X_train = training_data[feature_columns].values
y_train = training_data['LTPC'].values

#Initialize and train the model
parameters = {
    'n_estimators': 100,
    'max_features': 'sqrt',
    'max_depth': 10,
    'random_state': 456453,
    'criterion': 'gini'
}
model = RandomForestClassifier(**parameters)
model.fit(X_train, y_train)

#Save out model
import pickle

filename = 'results\\saved_models\\random_forest_model.pkl'
pickle.dump(model, open(filename, 'wb'))

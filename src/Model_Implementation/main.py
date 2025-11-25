'''
Implements a model to detect former farm land (maybe) using sklearn's RandomForestClassifier
'''

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
rand_seed = 2390

#Read in training data
training_data = pd.read_csv("data\\Training_Data.csv")

#Prepare feature matrix and labels
feature_columns = [col for col in training_data.columns if col not in ['ID', 'Northing', 'Easting', 'LTPC']]
X_train = training_data[feature_columns].values
y_train = training_data['LTPC'].values

#Create grid of parameters for grid search
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [5,10,15,20],
    'criterion': ['gini','entropy']
}
model = RandomForestClassifier(random_state=rand_seed)
grid_search = GridSearchCV(estimator=model, 
                           param_grid=param_grid, 
                           cv=5, 
                           n_jobs=-1, 
                           verbose=2,
                           scoring = 'roc_auc_ovr')
grid_search.fit(X_train, y_train)
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation ROC/AUC: ", grid_search.best_score_)
model = grid_search.best_estimator_

#Save out model
import pickle

filename = 'results\\saved_models\\random_forest_model.pkl'
pickle.dump(model, open(filename, 'wb'))

##Evaluate on test data
#Read in testing data
testing_data = pd.read_csv("data\\Testing_Data.csv")

#Prepare feature matrix and labels
feature_columns = [col for col in testing_data.columns if col not in ['ID', 'Northing', 'Easting', 'LTPC']]
X_test = training_data[feature_columns].values
y_test = training_data['LTPC'].values

#Make predictions
y_pred = model.predict(X_test)

#Generate confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=model.classes_)
disp.plot()

plt.title("Random Forest Classifier Confusion Matrix")
plt.savefig("results\\figures\\random_forest_confusion_matrix.png")
plt.show()

'''
Implements a model to detect former farm land (maybe)
'''

import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import modeling_tools as mt
rand_seed = 2390

#Dictionary of LTPC classes
lptc_dict = {
    1: "Stable pasture/hay",
    2: "Stable cropland",
    3: "Stable non-agricultural/non-developed",
    4: "Transitioned from pasture/hay to cropland",
    5: "Transitioned from pasture to non-agricultural/non-developed",
    6: "Transitioned from cropland to pasture",
    7: "Transitioned from cropland to NAND",
    8: "Transitioned from NAND to pasture",
    9: "Transitioned from NAND to crops"
}

#Read in training data
training_data = pd.read_csv("data\\Training_Data.csv")

#Prepare feature matrix and labels
feature_columns = [col for col in training_data.columns if col not in ['ID', 'Northing', 'Easting', 'LTPC']]
X_train = training_data[feature_columns].values
y_train = training_data['LTPC'].values

#Read in testing data
testing_data = pd.read_csv("data\\Testing_Data.csv")

#Prepare feature matrix and labels
feature_columns = [col for col in testing_data.columns if col not in ['ID', 'Northing', 'Easting', 'LTPC']]
X_test = testing_data[feature_columns].values
y_test = testing_data['LTPC'].values


##Random Forest model
#Create grid of parameters for grid search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['sqrt', 'log2'],
    'max_depth': [5,10,15,20],
    'criterion': ['gini','entropy']
}

rf_model, cm = mt.random_forest_model(X_train, y_train, X_test, y_test, param_grid, rand_seed)

#Save out model
import pickle
filename = 'results\\saved_models\\random_forest_model.pkl'
pickle.dump(rf_model, open(filename, 'wb'))

#Save out confusion matrix figure
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=rf_model.classes_)
disp.plot()
plt.title("Random Forest Classifier Confusion Matrix")
#Set x and y labels with class names
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
#plt.xticks(ticks=np.arange(len(rf_model.classes_)), labels=[lptc_dict.get(cls, str(cls)) for cls in rf_model.classes_], rotation=45, ha='right')
#plt.yticks(ticks=np.arange(len(rf_model.classes_)), labels=[lptc_dict.get(cls, str(cls)) for cls in rf_model.classes_])
plt.savefig("results\\figures\\random_forest_confusion_matrix.png")
plt.show()


##Support Vector Machine model
#Create grid of parameters for grid search
param_grid = {
    'C': [100, 1000, 10000, 100000],
    'gamma': [0.001, 0.01, 0.1, 1, 10, 100],
    'kernel': ['rbf']
}

sv_model, cm = mt.support_vector_model(X_train, y_train, X_test, y_test, param_grid, rand_seed)

#Save out model
import pickle
filename = 'results\\saved_models\\support_vector_model.pkl'
pickle.dump(sv_model, open(filename, 'wb'))

#Save out confusion matrix figure
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=sv_model.classes_)
disp.plot()
plt.title("Support Vector Classifier Confusion Matrix")
#Set x and y labels with class names
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
#plt.xticks(ticks=np.arange(len(rf_model.classes_)), labels=[lptc_dict.get(cls, str(cls)) for cls in rf_model.classes_], rotation=45, ha='right')
#plt.yticks(ticks=np.arange(len(rf_model.classes_)), labels=[lptc_dict.get(cls, str(cls)) for cls in rf_model.classes_])
plt.savefig("results\\figures\\support_vector_confusion_matrix.png")
plt.show()
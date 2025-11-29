'''
This code contains functions used to generate models as the final step in this project
'''
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd


def random_forest_model(X_train, y_train, X_test, y_test, param_grid, rand_seed = 0):
    '''
    Function to create and train a random forest model. Trains the model using a grid search and 5-fold cross validation.
    Returns the best model and a confusion matrix which can be used to evaluate the model.
    
    args:
        X_train: Training dataset features
        y_train: Training dataset labels
        X_test: Testing dataset features
        y_test: Testing dataset labels
        param_grid: Parameter grid used for grid search
        rand_seed: Optional random seed
    '''
    model = RandomForestClassifier(random_state=rand_seed)
    grid_search = GridSearchCV(estimator=model, 
                               param_grid=param_grid, 
                               cv=5, 
                               n_jobs=-1, 
                               verbose=2,
                               scoring = 'accuracy')
    grid_search.fit(X_train, y_train)
    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation accuracy: ", grid_search.best_score_)
    model = grid_search.best_estimator_

    #Make predictions
    y_pred = model.predict(X_test)

    #Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

    return model, cm


def support_vector_model(X_train, y_train, X_test, y_test, param_grid, rand_seed = 0):
    '''
    Function to create and train a support vector classifier model. Trains the model using a grid search and 5-fold cross validation.
    Returns the best model and a confusion matrix which can be used to evaluate the model.
    
    args:
        X_train: Training dataset features
        y_train: Training dataset labels
        X_test: Testing dataset features
        y_test: Testing dataset labels
        param_grid: Parameter grid used for grid search
        rand_seed: Optional random seed
    '''
    model = SVC(random_state=rand_seed)
    grid_search = GridSearchCV(estimator=model, 
                               param_grid=param_grid, 
                               cv=5, 
                               n_jobs=-1, 
                               verbose=2,
                               scoring = 'accuracy')
    grid_search.fit(X_train, y_train)
    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation accuracy: ", grid_search.best_score_)
    model = grid_search.best_estimator_

    #Make predictions
    y_pred = model.predict(X_test)

    #Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

    return model, cm
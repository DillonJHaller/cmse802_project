'''
This script performs SHAP analysis on trained models
'''
import pandas as pd
import numpy as np
import shap
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

#Load in saved pickle model
with open('results\\saved_models\\random_forest_model.pkl', 'rb') as f:
    rf_model = pickle.load(f)

importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
#Load in training data to get feature names
training_data = pd.read_csv("data\\Training_Data.csv")
feature_columns = [col for col in training_data.columns if col not in ['ID', 'Northing', 'Easting', 'LTPC']]

for i, idx in enumerate(indices[:10]):
    print(f"{i+1}. Feature: {feature_columns[idx]}, Importance: {importances[idx]}")

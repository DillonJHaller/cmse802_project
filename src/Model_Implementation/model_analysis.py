'''
This script performs analysis on trained models
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

#Basic plot of feature importances
plt.figure(figsize=(10,6))
plt.title("Feature Importances from Random Forest")
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), [feature_columns[i] for i in indices], rotation=90)
plt.tight_layout()
plt.savefig("results\\figures\\Model_analysis\\rf_feature_importances.png")

#Create shap explainer
explainer = shap.TreeExplainer(rf_model)
#Use a subset of training data for SHAP values
X_train = training_data[feature_columns].sample(n=100, random_state=42)
shap_values = explainer.shap_values(X_train)
#Summary plot of SHAP values for the ten most important features (Make the figure wider for readability)
plt.figure(figsize=(10,6))
shap.summary_plot(shap_values, X_train, feature_names=feature_columns, max_display=10, show=False, plot_size=(10,6), color_bar=True)
plt.savefig("results\\figures\\Model_analysis\\rf_shap_summary.png")


##Now the Support Vector Machine model
#Load in saved pickle model
with open('results\\saved_models\\support_vector_model.pkl', 'rb') as f:
    sv_model = pickle.load(f)

#Since SVM does not have built-in feature importances, we use SHAP values directly
explainer_sv = shap.KernelExplainer(sv_model.predict, shap.sample(X_train, 50))
shap_values_sv = explainer_sv.shap_values(X_train, nsamples=100)
#Summary plot of SHAP values for the ten most important features (Make the figure wider for readability)
plt.figure(figsize=(10,6))
shap.summary_plot(shap_values_sv, X_train, feature_names=feature_columns, max_display=10, show=False, plot_size=(10,6), color_bar=True)
plt.title("SHAP Summary Plot for Support Vector Machine")
plt.savefig("results\\figures\\Model_analysis\\svm_shap_summary.png")
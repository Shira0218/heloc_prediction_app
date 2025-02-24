#!/usr/bin/env python
# coding: utf-8

# ## ML GROUP PROJECT 2025
# 
# - Ksenia
# - Anh
# - Shuya
# - Vivi

# In[2]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[4]:


import subprocess
import sys

#
subprocess.check_call([sys.executable, "-m", "pip", "install", "imbalanced-learn"])



# In[6]:


# loading dataset
data_dictionary = "ML project 2025/heloc_data_dictionary-2.xlsx"

df = pd.read_csv('heloc_dataset_v1.csv')
df.head()


# In[8]:


from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from collections import Counter


# In[9]:


# check the missing values
n_cols_with_missing_values = df.isnull().any(axis=0).sum()

n_rows_with_missing_ExternalRiskEstimate_values = df['ExternalRiskEstimate'].isnull().sum()

print(n_cols_with_missing_values)
print(n_rows_with_missing_ExternalRiskEstimate_values)


# In[12]:


col_names = df.columns.tolist()

cols_numeric = [cols for cols in col_names if df[cols].dtype in ['int64', 'float64']]
cols_string = [cols for cols in col_names if df[cols].dtype == 'object']

print(cols_numeric)
print(cols_string)


# In[14]:


df = df[~df.isin([-9]).any(axis=1)]


# In[16]:


group_mean_7 = df.replace(-7, np.nan).groupby('RiskPerformance').mean()
group_mean_7


# In[18]:


# Function to impute -7 with group means based on Risk_Performance
def impute_with_group_mean(row):
    for col in df.columns:
        if col == 'RiskPerformance' or not np.issubdtype(df[col].dtype, np.number):
            continue
        # Replace -7 with the corresponding group's mean
        if row[col] == -7:
            row[col] = group_mean_7.loc[row['RiskPerformance'], col]
    return row

# Apply the function row by row
df = df.apply(impute_with_group_mean, axis=1)


# In[19]:


duplicates = df.duplicated()
num_duplicates = duplicates.sum()
num_duplicates


# In[20]:


duplicate_rows = df[duplicates]
duplicate_rows


# In[21]:


df = df.drop_duplicates()


# In[26]:


label_encoder = LabelEncoder()

# Separate features and target variable
selected = ['MSinceMostRecentDelq', 'MaxDelqEver', 'ExternalRiskEstimate', 'PercentTradesNeverDelq','MSinceMostRecentInqexcl7days']
X = df[selected]
#X = df.drop(columns=['RiskPerformance'])
y = label_encoder.fit_transform(df['RiskPerformance'])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[28]:


# Apply Lasso regression with cross-validation
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_scaled, y)


# In[30]:


# Extract feature importance
lasso_importance = pd.Series(lasso.coef_, index=X.columns)
important_features = lasso_importance[lasso_importance != 0].sort_values(ascending=False)
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_scaled, y)


# In[32]:


# Extract feature importance
lasso_importance = pd.Series(lasso.coef_, index=X.columns)


# In[34]:


# Print the top 10 most important features
top_n = 24
print(f"Top {top_n} Important Features Based on Lasso Regression:")
print(important_features.head(top_n))


# In[36]:


# Class Imbalance
counts = df["RiskPerformance"].value_counts()
print("Counts of 'Bad' vs 'Good':")
print(counts)


# In[38]:


from sklearn.model_selection import train_test_split

# Step 2: Split the data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

smote = SMOTE(random_state=42)
X_train_s, y_train_s = smote.fit_resample(X_train, y_train)

# New class distribution
print("Resampled class distribution:", Counter(y_train_s))


# In[40]:


rows=df.shape[0]
rows


# In[42]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb


# ### Model training and evaluation
# 
# After preprocessed the data, now we can start training 3 models: classification tree, logistic regression, and K-nearest neighbors.
# 
# We will use 2 methods for evaluation: validation set as simple cross validation and cross-validation

# Logistic Regression Model

# In[46]:


log_reg=LogisticRegression(solver='liblinear', random_state=42)
log_reg.fit(X_train_s, y_train_s)

# Make predictions
y_pred = log_reg.predict(X_test)


# In[48]:


log_accuracy = accuracy_score(y_test, y_pred)
log_precision = precision_score(y_test, y_pred)
log_recall = recall_score(y_test, y_pred)
log_f1 = f1_score(y_test, y_pred)

print("🔍 Model Performance Metrics:")
print(f"Accuracy: {log_accuracy:.4f}")
print(f"Precision: {log_precision:.4f}")
print(f"Recall: {log_recall:.4f}")
print(f"F1 Score: {log_f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# In[50]:


# Visualization
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Bad (0)", "Good (1)"], yticklabels=["Bad (0)", "Good (1)"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# ### Decision Tree

# In[53]:


# Do the split once and capture all outputs
X_train_t_tr, X_train_t_val, Y_train_t_tr, Y_train_t_val = train_test_split(
    X_train_s, 
    y_train_s, 
    test_size=0.25, 
    random_state=42,
    stratify=y_train_s
)

# Verify the shapes
print('Training set shape:', X_train_t_tr.shape)
print('Validation set shape:', X_train_t_val.shape)
print('Training labels shape:', Y_train_t_tr.shape)
print('Validation labels shape:', Y_train_t_val.shape)

# Verify the proportions
print('\nSplit proportions:')
total_samples = len(X_train_s)
print(f'Training: {len(X_train_t_tr)/total_samples:.2f}')
print(f'Validation: {len(X_train_t_val)/total_samples:.2f}')


# In[55]:


# check the shape

print('The shape of X_train_t_tr:', X_train_t_tr.shape)
print('The shape of X_train_t_val', X_train_t_val.shape)
print('The shape of Y_train_t_tr:', Y_train_t_tr.shape)
print('The shape of Y_train_t_val:', Y_train_t_val.shape)


# Validation set approach: 
# 
# - We use 60% of the data to train models (X_train_t_tr, Y_train_t_tr)
# - We use 20% of the data for model selection (X_train_t_val, Y_train_t_val)
# - We use the remaining 20% for final evaluation (X_test_transformed, Y_test)

# In[58]:


from sklearn import tree, linear_model, neighbors

clf_tree = tree.DecisionTreeClassifier().fit(X_train_t_tr, Y_train_t_tr)
clf_log_reg = linear_model.LogisticRegression(max_iter=10000).fit(X_train_t_tr, Y_train_t_tr)
clf_knn = neighbors.KNeighborsClassifier().fit(X_train_t_tr, Y_train_t_tr)


# In[60]:


from sklearn.metrics import accuracy_score

print('Decision tree accuracy: %3f'%accuracy_score(Y_train_t_val, clf_tree.predict(X_train_t_val)))
print('Logistic regression accuracy: %3f'%accuracy_score(Y_train_t_val, clf_log_reg.predict(X_train_t_val)))
print('KNN accuracy: %.3f'%accuracy_score(Y_train_t_val, clf_knn.predict(X_train_t_val)))


# In[62]:


from sklearn.model_selection import cross_validate
from sklearn import tree, linear_model, neighbors

cv_result_tree = cross_validate(tree.DecisionTreeClassifier(), X_train_s, y_train_s, cv=5, return_estimator=True)
cv_result_log_reg = cross_validate(linear_model.LogisticRegression(max_iter=10000), X_train_s, y_train_s, cv=5, return_estimator=True)
cv_result_knn = cross_validate(neighbors.KNeighborsClassifier(), X_train_s, y_train_s, cv=5, return_estimator=True)


# In[63]:


cv_result_tree


# In[64]:


# this the average value of each model
print('Classification tree - CV accuracy score is %.3f'%cv_result_tree['test_score'].mean())
print('Logistic regression - CV accuracy score %.3f'%cv_result_log_reg['test_score'].mean())
print('KKN - CV accuracy score %.3f'%cv_result_knn['test_score'].mean())


# In[65]:


from sklearn.model_selection import GridSearchCV

param_grid = [{'max_depth':[1,2,3,4,5],  
               'criterion':["gini", "entropy"],            
               'min_samples_split':[2,5,10],              
               'min_samples_leaf':[10,20,30]
}]

clf_tree = tree.DecisionTreeClassifier()
grid_search = GridSearchCV(clf_tree, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train_s, y_train_s)


# In[66]:


cv_res = grid_search.cv_results_
for mean_score, params in zip(cv_res['mean_test_score'], cv_res['params']):
    print(mean_score, params)


# In[67]:


# the best hyper-parameters we found so far

grid_search.best_params_


# In[68]:


# variable holding the best classifier (fitted on the entire dataset)

grid_search.best_estimator_


# - Now we are training a model on the train set to evaluate on the validation set

# In[73]:


clf_tree = tree.DecisionTreeClassifier(max_depth=5,min_samples_leaf=10).fit(X_train_t_tr, Y_train_t_tr)


# In[74]:


from sklearn.metrics import confusion_matrix

y_pred = clf_tree.predict(X_train_t_val)
conf_matrix = confusion_matrix(Y_train_t_val, y_pred)

print(conf_matrix)


# In[75]:


from sklearn.metrics import accuracy_score, recall_score, precision_score

tn, fp, fn, tp = conf_matrix[0,0], conf_matrix[0,1], conf_matrix[1,0], conf_matrix[1,1]

accuracy = (tn + tp)/ (tn + tp + fn + fp)

tpr = tp/ (tp + fn)
fpr = fp/ (fp + tn)
tnr = tn/ (tn + fp)
fnr = fn/ (fn + tp)

recall = recall_score(Y_train_t_val, clf_tree.predict(X_train_t_val))
precision = precision_score(Y_train_t_val, clf_tree.predict(X_train_t_val))

print('tn is %.3f'%tn)
print('fp is %.3f'%fp)
print('fn is %.3f'%fn)
print('tp is %.3f'%tp)
print('tpr is %.3f'%tpr)
print('fpr is %f'%fpr)
print('tnr is %f'%tnr)
print('fnr is %f'%fnr)
print('Recall is %.3f'%recall)
print('Precision is %f'%precision)
print('Accuracy is %f'%accuracy)
# Additional metrics that might be useful
f1_score = 2 * (precision * tpr) / (precision + tpr)
print(f'F1 Score is {f1_score:.3f}')


# In[83]:


# Plotting the ROC curve for this model

from sklearn import metrics

scores = clf_tree.predict_proba(X_train_t_val)[:,1]
fpr, tpr, thresholds = metrics.roc_curve(Y_train_t_val, scores)
auc = metrics.auc(fpr, tpr)

plt.figure(figsize=(5,5))
lw = 2

idx = (np.abs(thresholds - 0.5)).argmin()
selected_fpr, selected_tpr = fpr[idx], tpr[idx]

plt.plot(fpr, tpr, color='darkorange', lw=lw, label = 'ROC curve (area = %0.2f)' % auc)
plt.plot([0,1], [0,1], color= 'navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right');

plt.plot(selected_fpr, selected_tpr, marker='x', markeredgewidth=5, markersize=12);


# - ROC curve shows a series of models that differ by certain threshold value (on the probability for predicting 1 vs 0)
# - 'X' marks the trained model clf_tree

# ### Using Tree-based models

# In[87]:


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold


# In[89]:


# Using the decision stumps (max_depth=1)

clf_tree_1 = DecisionTreeClassifier(max_depth=1, random_state=0).fit(X_train_t_tr, Y_train_t_tr)


# In[91]:


train_accuracy_dt = clf_tree_1.score(X_train_t_tr, Y_train_t_tr)
val_accuracy_dt = clf_tree_1.score(X_train_t_val, Y_train_t_val)

print('Train accuracy (DT): %.3f'%train_accuracy_dt)
print('Validation accuracy (DT): %.3f'%val_accuracy_dt)


# In[93]:


# changing the depth of tree

accuracy_tr = {'Train accuracy':[], 'Validation accuracy':[]}

for depth in range (1,13):
    clf_tr = DecisionTreeClassifier(max_depth= depth, random_state=0)
    clf_tr.fit(X_train_t_tr, Y_train_t_tr)

    acc_train = clf_tr.score(X_train_t_tr, Y_train_t_tr)
    acc_val = clf_tr.score(X_train_t_val, Y_train_t_val)

    accuracy_tr['Train accuracy'].append(acc_train)
    accuracy_tr['Validation accuracy'].append(acc_val)

tree_accuracy = pd.DataFrame(accuracy_tr, index= range(1,13))

print(tree_accuracy.head())
print(tree_accuracy.plot())


# Based on this plotting of the train and validation accuracies as the tree depth, we can conclude that there is no improvement of the validation accuracy (shown in the deterioration at some point) in the meanwhile the train accuracy keep improving. ==> This is an indication of overfitting.
# 
# - Now, we are trying to tune the hyperparameters of the model to improve its performance.

# In[96]:


param_grid_1 = [{'max_depth': range(1,13),
                 'min_samples_leaf': [10,20,100],
                 'max_leaf_nodes': [2,4,6,20,100,10000]}]

grid_search_1 = GridSearchCV(DecisionTreeClassifier(random_state=0),
                             param_grid_1,
                             cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
                             scoring='accuracy')

grid_search_1.fit(X_train_t_tr, Y_train_t_tr)


# In[97]:


# Creating the datafram based on the results of the grid search above

grid_search_1_res = grid_search_1.cv_results_

tree_accuracy_grid = pd.DataFrame({
    'max_depth': grid_search_1_res['param_max_depth'].data,
    'max_leaf_nodes': grid_search_1_res['param_max_leaf_nodes'].data,
    'min_samples_leaf': grid_search_1_res['param_min_samples_leaf'].data,
    'Accuracy': grid_search_1_res['mean_test_score']
})

print(tree_accuracy_grid.head())


# In[98]:


import seaborn as sns 

fig, axes = plt.subplots(1,2, figsize=(20,5))
plt.suptitle('Decision trees: CV Accuracy vs Depth \n(Distribution over hyperparameters that share the same tree depth)')
sns.boxplot(x='max_depth', y='Accuracy', data=tree_accuracy_grid, ax=axes[0], palette='coolwarm');
tree_accuracy.plot(ax=axes[1]);


# The figure above shows the cross validation accuracy as a function of the maximal depth. The right tuning can miltigate overfitting.

# ### Using Ensemble methods - Random forests model

# In[101]:


rf = RandomForestClassifier(max_depth=1, random_state=0).fit(X_train_s, y_train_s)
rf


# In[102]:


random_grid = {
    'n_estimators': np.arange(100, 1000, 100),
    'max_depth': np.arange(10, 50, 5),
    'min_samples_split': np.arange(2, 10, 2),
    'min_samples_leaf': np.arange(1, 5, 1),
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', 'balanced_subsample']}


# In[108]:


random_search = RandomizedSearchCV(rf, param_distributions=random_grid, 
                                   n_iter=20, cv=5, scoring='f1', n_jobs=-1, verbose=2, random_state=42)
random_search.fit(X_train_s, y_train_s)


# In[109]:


best_rf_model_random = random_search.best_estimator_
best_rf_model_random


# In[110]:


from sklearn.metrics import f1_score


# Predictions on Test Set
y_pred_rf_random = best_rf_model_random.predict(X_test)

# Evaluate Performance
accuracy_random = accuracy_score(y_test, y_pred_rf_random)
precision_random = precision_score(y_test, y_pred_rf_random)
recall_random = recall_score(y_test, y_pred_rf_random)
f1_random = f1_score(y_test, y_pred_rf_random)

print("Best Hyperparameters (via Random Search):", random_search.best_params_)
print("Tuned Random Forest Performance (Randomized Search):")
print(f"Accuracy: {accuracy_random:.4f}")
print(f"Precision: {precision_random:.4f}")
print(f"Recall: {recall_random:.4f}")
print(f"F1 Score: {f1_random:.4f}")


# In[111]:


#Confusion Matrix Visualization
cm = confusion_matrix(y_test, y_pred_rf_random)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Bad (0)", "Good (1)"], yticklabels=["Bad (0)", "Good (1)"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# As the graph above is given, we observe that the performance of the ensemble tends to improve and stablize as the ensembe grows.

# ### Using Ensemble method - Boosting model

# In[114]:


clf_boosting = AdaBoostClassifier(random_state=0).fit(X_train_s, y_train_s)
clf_boosting


# In[115]:


train_boosting_acc = clf_boosting.score(X_train_s, y_train_s)
val_boosting_acc = clf_boosting.score(X_train_s, y_train_s)


# In[116]:


print('Train accuracy (boosting): %.3f'%train_boosting_acc)
print('Validation accuracy (boosting): %.3f'%val_boosting_acc)


# Now, we are testing the effect of the learning rate with higher learning rate and a larger ensemble because the smaller the learning rate, the more iterations we may need to reach the optimal ensemble.

# In[129]:


# Initialize AdaBoost with a base Decision Tree (stump)
b_estimator = DecisionTreeClassifier(max_depth=1)
clf_boosting = AdaBoostClassifier(estimator=b_estimator, n_estimators=50, learning_rate=1.0, random_state=42)

# Train the model
clf_boosting.fit(X_train_s, y_train_s)

# Predictions
y_train_pred = clf_boosting.predict(X_train_s)
y_test_pred = clf_boosting.predict(X_test)

# Performance Metrics
train_acc = accuracy_score(y_train_s, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
train_f1 = f1_score(y_train_s, y_train_pred)
test_f1 = f1_score(y_test, y_test_pred)

print(f"Train Accuracy: {train_acc:.3f}")
print(f"Test Accuracy: {test_acc:.3f}")
print(f"Train F1 Score: {train_f1:.3f}")
print(f"Test F1 Score: {test_f1:.3f}")

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_test_pred))

# Feature Importance
feature_importance = clf_boosting.feature_importances_
print("Feature Importance:\n", feature_importance)

# Hyperparameter tuning with GridSearchCV
param_grid = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.1, 1.0],
    "estimator__max_depth": [1, 2, 3]  # Tuning weak learner depth
}

grid_search = GridSearchCV(AdaBoostClassifier(estimator=DecisionTreeClassifier()), param_grid, cv=5, scoring="f1")
grid_search.fit(X_train_s, y_train_s)

print("Best Hyperparameters:", grid_search.best_params_)
print("Best F1 Score:", grid_search.best_score_)


# In[130]:


# Predictions on test data
y_test_pred_ada = clf_boosting.predict(X_test)
accuracy_boosting = accuracy_score(y_test, y_test_pred)
precision_boosting = precision_score(y_test, y_test_pred)
recall_boosting = recall_score(y_test, y_test_pred)
f1_boosting = grid_search.best_score_
print(f"Accuracy (Boosting): {accuracy_boosting:.4f}")
print(f"Precision (Boosting): {precision_boosting:.4f}")
print(f"Recall (Boosting): {recall_boosting:.4f}")
print(f"Best F1 Score: {f1_boosting:.4f}")


# In[131]:


# Compute Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)

# Plot the Confusion Matrix
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Bad (0)", "Good (1)"], yticklabels=["Bad (0)", "Good (1)"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


# In[132]:


# Extract feature importance from AdaBoost
feature_importance = clf_boosting.feature_importances_

# Create a Pandas Series with feature names
ada_feature_importance = pd.Series(feature_importance, index=X_train.columns)

# Select top 10 most important features
top_features_ada = ada_feature_importance.sort_values(ascending=False).head(10)

# Plot the feature importance
plt.figure(figsize=(8, 5))
sns.barplot(x=top_features_ada.values, y=top_features_ada.index, palette="viridis")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Top 10 Most Important Features - AdaBoost")
plt.show()


# XGBoost

# In[134]:


# Train XGBoost Model
xgb_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    use_label_encoder=False,
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,  # Adjust this for imbalanced data
    random_state=42
)

xgb_model.fit(X_train_s, y_train_s)



# In[135]:


# Make Predictions
y_pred_xgb = xgb_model.predict(X_test)

# Model Evaluation
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
precision_xgb = precision_score(y_test, y_pred_xgb)
recall_xgb = recall_score(y_test, y_pred_xgb)
f1_xgb = f1_score(y_test, y_pred_xgb)

print("XGBoost Model Performance:")
print(f"Accuracy: {accuracy_xgb:.4f}")
print(f"Precision: {precision_xgb:.4f}")
print(f"Recall: {recall_xgb:.4f}")
print(f"F1 Score: {f1_xgb:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred_xgb))


# In[136]:


#Confusion Matrix
cm_xgb = confusion_matrix(y_test, y_pred_xgb)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Blues", xticklabels=["Bad (0)", "Good (1)"], yticklabels=["Bad (0)", "Good (1)"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - XGBoost")
plt.show()


# In[137]:


#Feature Importance Visualization
xgb_feature_importance = pd.Series(xgb_model.feature_importances_, index=selected)
top_features_xgb = xgb_feature_importance.sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
sns.barplot(x=top_features_xgb.values, y=top_features_xgb.index, palette="viridis")
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Top 10 Most Important Features - XGBoost")
plt.show()


# In[138]:


model_comparison = {
    "Model": ["XGBoost", "AdaBoost", "Random Forest (Tuned)", "Logistic Regression"],
    "Accuracy": [accuracy_xgb, accuracy_boosting, accuracy_random, log_accuracy],
    "Precision": [precision_xgb, precision_boosting, precision_random, log_precision],
    "Recall": [recall_xgb, recall_boosting, recall_random, log_recall],
    "F1-Score": [f1_xgb, f1_boosting, f1_random, log_f1]
}

# Convert to DataFrame
comparison_df = pd.DataFrame(model_comparison)
comparison_df


# In[155]:


import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 生成示例数据集
X, y = make_classification(n_samples=100, n_features=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 保存模型到桌面路径
desktop_path = '/Users/shuyachen/Desktop/model.pkl'  # 将路径修改为你的桌面路径
with open(desktop_path, 'wb') as f:
    pickle.dump(model, f)

print(f"Model saved as {desktop_path}")


# In[159]:


import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Supported languages
languages = {
    'English': {
        'app_title': 'HELOC Eligibility Prediction App',
        'choose_file': 'Choose a file',
        'drag_drop': 'Drag and drop file here',
        'or_manual_input': 'Or manually input data',
        'submit': 'Submit',
        'prediction': 'Prediction Result',
        'accept': 'Accept',
        'reject': 'Reject',
        'reason_accept': 'Congratulations! Your application is accepted.',
        'reason_reject': 'Sorry, your application is rejected due to risk factors.'
    },
    'Chinese': {
        'app_title': 'HELOC 资格预测应用',
        'choose_file': '选择文件',
        'drag_drop': '将文件拖放到这里',
        'or_manual_input': '或者手动输入数据',
        'submit': '提交',
        'prediction': '预测结果',
        'accept': '接受',
        'reject': '拒绝',
        'reason_accept': '恭喜！您的申请已被接受。',
        'reason_reject': '抱歉，由于风险因素，您的申请被拒绝。'
    },
    'Russian': {
        'app_title': 'Приложение для прогнозирования HELOC',
        'choose_file': 'Выберите файл',
        'drag_drop': 'Перетащите файл сюда',
        'or_manual_input': 'Или введите данные вручную',
        'submit': 'Отправить',
        'prediction': 'Результат прогноза',
        'accept': 'Принято',
        'reject': 'Отклонено',
        'reason_accept': 'Поздравляем! Ваша заявка принята.',
        'reason_reject': 'Извините, ваша заявка отклонена из-за факторов риска.'
    },
    'Vietnamese': {
        'app_title': 'Ứng dụng Dự đoán Đủ điều kiện HELOC',
        'choose_file': 'Chọn tệp',
        'drag_drop': 'Kéo và thả tệp vào đây',
        'or_manual_input': 'Hoặc nhập dữ liệu thủ công',
        'submit': 'Gửi',
        'prediction': 'Kết quả Dự đoán',
        'accept': 'Chấp nhận',
        'reject': 'Từ chối',
        'reason_accept': 'Chúc mừng! Đơn đăng ký của bạn đã được chấp nhận.',
        'reason_reject': 'Xin lỗi, đơn đăng ký của bạn đã bị từ chối do các yếu tố rủi ro.'
    },
    'Hindi': {
        'app_title': 'HELOC पात्रता भविष्यवाणी ऐप',
        'choose_file': 'फ़ाइल चुनें',
        'drag_drop': 'फ़ाइल को यहाँ खींचें और छोड़ें',
        'or_manual_input': 'या मैन्युअल रूप से डेटा इनपुट करें',
        'submit': 'जमा करें',
        'prediction': 'भविष्यवाणी परिणाम',
        'accept': 'स्वीकार करें',
        'reject': 'अस्वीकार करें',
        'reason_accept': 'बधाई हो! आपका आवेदन स्वीकार कर लिया गया है।',
        'reason_reject': 'क्षमा करें, आपके आवेदन को जोखिम कारकों के कारण अस्वीकार कर दिया गया है।'
    }
}

# Streamlit app layout
st.sidebar.title('Select Language')
language = st.sidebar.selectbox('Select Language', list(languages.keys()))
text = languages[language]

st.title(text['app_title'])

uploaded_file = st.file_uploader(text['choose_file'], type=['csv', 'xlsx'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
    predictions = model.predict(df)
    st.subheader(text['prediction'])
    for pred in predictions:
        if pred == 1:
            st.success(f"{text['accept']}: {text['reason_accept']}")
        else:
            st.error(f"{text['reject']}: {text['reason_reject']}")
else:
    st.write(text['or_manual_input'])
    manual_input = {}
    input_columns = ['ExternalRiskEstimate', 'MSinceOldestTradeOpen', 'MSinceMostRecentTradeOpen']
    for col in input_columns:
        manual_input[col] = st.number_input(label=col, value=0)
    
    if st.button(text['submit']):
        input_data = pd.DataFrame([manual_input])
        predictions = model.predict(input_data)
        st.subheader(text['prediction'])
        for pred in predictions:
            if pred == 1:
                st.success(f"{text['accept']}: {text['reason_accept']}")
            else:
                st.error(f"{text['reject']}: {text['reason_reject']}")



# In[ ]:





# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 15:58:26 2023

@author: avaz5
"""
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, auc, RocCurveDisplay, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.feature_selection import SelectFromModel
from xgboost import XGBClassifier
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE

df = pd.read_csv('Base.csv')

#%%
# Check fraud rate with rows with -1s removed in variables that use as an imputed value for nulls.
imputes_df = df[(df['prev_address_months_count'] != -1) & (df['current_address_months_count'] != -1) & (df['bank_months_count'] != -1)]
print(f'Original fraud rate: {df["fraud_bool"].mean():.3f}\nImputes removed fraud rate: {imputes_df["fraud_bool"].mean():.3f}')
# Nearly 5 times less (0.011 vs 0.002). Will just remove columns instead.

cols_with_imputes = ['bank_months_count', 'prev_address_months_count', 'current_address_months_count']

df.drop(columns = cols_with_imputes, inplace = True)

#%%
# Prepare data for reuse.
# Categorize variable types.

nominal_cols = ['payment_type','employment_status','email_is_free','housing_status',
                    'phone_home_valid','phone_mobile_valid','has_other_cards',
                    'foreign_request','source','device_os','keep_alive_session',
                    'customer_age']

ordinal_col = 'income'

numeric_cols = ['name_email_similarity','days_since_request',
                'intended_balcon_amount','zip_count_4w','velocity_6h','velocity_24h',
                'velocity_4w','bank_branch_count_8w','date_of_birth_distinct_emails_4w',
                'credit_risk_score','proposed_credit_limit','session_length_in_minutes',
                'device_distinct_emails_8w', 'device_fraud_count']

categorical_cols = nominal_cols + [ordinal_col]

# Convert nominal categorical variables to string type since some have values represented as integers
df[nominal_cols] = df[nominal_cols].astype(str)

# Income in this case was ordinal (deciles), let's appropriately transform this column
df[ordinal_col] = df[ordinal_col].astype(str)
df[ordinal_col] = df[ordinal_col].apply(lambda x: x[:3] if len(x) > 3 else x[:3])

#%%
# Prepare data for analysis - dummies, encoding, and standardizing
initial_df = df.copy()

# Ordinal encode income
ordinal_encoder = OrdinalEncoder()
# Need to specify 2D array (in this case we pass a dataframe)
initial_df[ordinal_col] = ordinal_encoder.fit_transform(initial_df[[ordinal_col]])
initial_df = pd.get_dummies(initial_df, columns = nominal_cols, drop_first = True)


# Split based on the month - this is a common practice!
initial_train_df = initial_df[initial_df['month'] < 6].copy()
initial_test_df = initial_df[initial_df['month'] >= 6].copy()

initial_train_df.drop(columns = 'month', inplace = True)
initial_test_df.drop(columns = 'month', inplace = True)

X_initial_train = initial_train_df.iloc[:, 1:]
y_initial_train = initial_train_df['fraud_bool']
X_initial_test = initial_test_df.iloc[:, 1:]
y_initial_test = initial_test_df['fraud_bool']

# We should actually split our data first and then standardize.
# https://datascience.stackexchange.com/a/54909
# https://www.baeldung.com/cs/data-normalization-before-after-splitting-set
# Standardize numeric columns, but fit only on the training set to avoid data leakage w/test set.
# X_initial_train, X_initial_test, y_initial_train, y_initial_test = train_test_split(X_initial, 
#                                                                                     y_initial, test_size = 0.2,
#                                                                                     random_state = 1)

scaler = StandardScaler()
X_initial_train[numeric_cols] = scaler.fit_transform(X_initial_train[numeric_cols])
X_initial_test[numeric_cols] = scaler.transform(X_initial_test[numeric_cols])

#%%
# Models fit on ALL variables
# First going to fit a regular logistic regression.
# Need to reduce the amount of features we have
# Helper function we can use for our simple models (no parameter tuning).

def train_and_eval_model(name, model, X_train, y_train, X_test, y_test):
    classifier = model.fit(X_train, y_train)
    print(f"The {name} model's accuracy on the training set is: {classifier.score(X_train, y_train):.2f}")
    
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)[: , 1]
    print(f"The {name} model's accuracy on the test set is {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))
    
    # Confusion matrix plotted
    disp = ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred))
    disp.plot()
    disp.ax_.set_title(f"Confusion matrix for {name} model")
    
    
    # Create roc_curve. Must first calculate the components
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    # Calculate AUC
    auc_calc = auc(fpr, tpr)
    # Build the plot
    roc = RocCurveDisplay(fpr = fpr, tpr = tpr, roc_auc = auc_calc)
    roc.plot()
    roc.ax_.set_title(f"ROC Curve for {name} model")
    
# Needed to increase the max_iter value to 500 in order for the logistic model to converge

initial_logistic = LogisticRegression(max_iter = 500)
initial_tree = DecisionTreeClassifier()
initial_forest = RandomForestClassifier()

models = [('Logistic', initial_logistic), 
          ('Decision Tree', initial_tree), 
          ('Random Forest', initial_forest)]

for name, model in models:
    train_and_eval_model(name, model, X_initial_train, y_initial_train, X_initial_test, y_initial_test)


#%%
# Perform a Lasso regression to identify selected features that we can use in regular logistic regression
# https://datascience.stackexchange.com/a/75385
# Example from https://www.blog.trainindata.com/lasso-feature-selection-with-python/
lasso_df = df.copy()

# First ordinal encode all categorical variables. Various sites suggested trying this in the feature selection phase.
ordinal_encoder2 = OrdinalEncoder()
# Get all categorical variables
categorical_cols = nominal_cols + [ordinal_col]
lasso_df[categorical_cols] = ordinal_encoder2.fit_transform(lasso_df[categorical_cols])

lasso_train_df = lasso_df[lasso_df['month'] < 6].copy()
lasso_test_df = lasso_df[lasso_df['month'] >= 6].copy()

lasso_train_df.drop(columns = 'month', inplace = True)
lasso_test_df.drop(columns = 'month', inplace = True)

X_lasso_train = lasso_train_df.iloc[:, 1:]
y_lasso_train = lasso_train_df['fraud_bool']
X_lasso_test = lasso_test_df.iloc[:, 1:]
y_lasso_test = lasso_test_df['fraud_bool']

# Standardize numeric columns
scaler = StandardScaler()
X_lasso_train[numeric_cols] = scaler.fit_transform(X_lasso_train[numeric_cols])
X_lasso_test[numeric_cols] = scaler.transform(X_lasso_test[numeric_cols])

# Create our model.
sel_ = SelectFromModel(LogisticRegression(C = 0.5, penalty = 'l1', solver = 'liblinear', random_state = 1))
sel_.fit(X_lasso_train, y_lasso_train)

print(sel_.get_support())

col_selection = pd.DataFrame({'col': X_lasso_train.columns, 'keep?': sel_.get_support()})

#%%
# Device fraud count was the only one not kept. 
# Let's now retry with that variable removed, but using dummy variables again
# Reusing intial_df from first series of models.

X_new_logistic_train = X_initial_train.copy()
X_new_logistic_test = X_initial_test.copy()
y_new_logistic_train = y_initial_train.copy()
y_new_logistic_test = y_initial_test.copy()

X_new_logistic_train.drop(columns = 'device_fraud_count', inplace = True)
X_new_logistic_test.drop(columns = 'device_fraud_count', inplace = True)

logistic_revised_1 = LogisticRegression(max_iter = 500)

# Use helper function created above.
train_and_eval_model('Logistic Revised 1', logistic_revised_1, X_new_logistic_train, y_new_logistic_train, 
                     X_new_logistic_test, y_new_logistic_test)

logistic_revised_1_odds_df = pd.DataFrame(np.exp(logistic_revised_1.coef_[0]), 
                                        index = X_new_logistic_train.columns, columns = ['Odds'])
print(logistic_revised_1_odds_df.sort_values(by='Odds', ascending = False))  

# Still overpredicting. Let's remove features based off exploratory analysis, numerical features 
# which had less than a 25% difference in mean between the different values for fraud_bool.
# They were: days_since_request, zip_count_4w, velocity_6h, velocity_24h, velocity_4w, 
# and session_length_in_minutes.

cols_to_remove = ['days_since_request', 'zip_count_4w', 'velocity_6h', 'velocity_24h','velocity_4w',
                  'session_length_in_minutes']
X_new_logistic_train.drop(columns = cols_to_remove, inplace = True)
X_new_logistic_test.drop(columns = cols_to_remove, inplace = True)

logistic_revised_2 = LogisticRegression(max_iter = 500)
train_and_eval_model('Logistic Revised 2', logistic_revised_2, X_new_logistic_train, y_new_logistic_train, 
                     X_new_logistic_test, y_new_logistic_test)

logistic_revised_2_odds_df = pd.DataFrame(np.exp(logistic_revised_2.coef_[0]), 
                                        index = X_new_logistic_train.columns, columns = ['Odds'])
print(logistic_revised_2_odds_df.sort_values(by='Odds', ascending = False))  

sm = SMOTE(random_state = 1)
X_new_logistic_train, y_new_logistic_train = sm.fit_resample(X_new_logistic_train, y_new_logistic_train)

# logistic_balanced = LogisticRegression(max_iter = 500, class_weight = 'balanced')
logistic_balanced = LogisticRegression(max_iter = 500)
train_and_eval_model('Logistic Balanced', logistic_balanced, X_new_logistic_train, y_new_logistic_train,
                     X_new_logistic_test, y_new_logistic_test)

# still pretty bad...but we got some improvement

#%%
# Let's now try decision trees & random forests again, but with some slight changes
# We will use OrdinalEncoder and tune parameters such as max_depth to attempt to eliminate overfitting.
# https://www.youtube.com/watch?v=n_x40CdPZss

tree_models_df = df.copy()

ordinal_encoder3 = OrdinalEncoder()
tree_models_df[categorical_cols] = ordinal_encoder3.fit_transform(tree_models_df[categorical_cols])

tree_train_df = tree_models_df[tree_models_df['month'] < 6].copy()
tree_test_df = tree_models_df[tree_models_df['month'] >= 6].copy()

tree_train_df.drop(columns = 'month', inplace = True)
tree_test_df.drop(columns = 'month', inplace = True)

X_tree_train = tree_train_df.iloc[:, 1:]
y_tree_train = tree_train_df['fraud_bool']
X_tree_test = tree_test_df.iloc[:, 1:]
y_tree_test = tree_test_df['fraud_bool']

scaler3 = StandardScaler()
scaler3.fit_transform(X_tree_train[numeric_cols])
X_tree_test[numeric_cols] = scaler3.transform(X_tree_test[numeric_cols])

X_tree_train.drop(columns = 'device_fraud_count', inplace = True)
X_tree_test.drop(columns = 'device_fraud_count', inplace = True)

X_tree_train.drop(columns = cols_to_remove, inplace = True)
X_tree_test.drop(columns = cols_to_remove, inplace = True)

# Tree parameters we will test.
# tree_params = {'criterion': ['gini', 'entropy'], 'max_depth': np.arange(1, 21, 2),
#           'min_samples_split': np.arange(1, 41  , 2), 'class_weight':['balanced'], 'max_features': ['log2']}
tree_params = {'criterion': ['gini', 'entropy'], 'max_depth': np.arange(1, 21, 2),
          'min_samples_split': np.arange(1, 41, 2), 'max_features': ['log2']}

# grid_search_tree = GridSearchCV(DecisionTreeClassifier(class_weight = 'balanced'), 
#                                 tree_params, n_jobs= -1, scoring = 'roc_auc')

grid_search_tree = GridSearchCV(DecisionTreeClassifier(), 
                                tree_params, n_jobs= -1, scoring = 'roc_auc')


sm2 = SMOTE(random_state = 1)
X_tree_train, y_tree_train = sm2.fit_resample(X_tree_train, y_tree_train)

grid_search_tree.fit(X_tree_train, y_tree_train) 
 
train_and_eval_model('Grid Search - Tree', grid_search_tree, X_tree_train, y_tree_train,
                     X_tree_test, y_tree_test)

print(grid_search_tree.best_estimator_)


#%%
# Trying xgboost
# https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
# https://machinelearningmastery.com/tune-number-size-decision-trees-xgboost-python/
# https://towardsdatascience.com/beginners-guide-to-xgboost-for-classification-problems-50f75aac5390

# Need to add class_weight balanced
# classes_weights = class_weight.compute_sample_weight(class_weight = 'balanced', y = y_tree_train)

bst = XGBClassifier()

xgb_params = {"n_estimators": np.arange(50, 101, 50) , "max_depth": np.arange(3, 11, 2), 
              "learning_rate": np.arange(0.05, 0.21, 0.05),
              "colsample_bytree": [0.8], 'seed': [1]}

grid_search_xgb = GridSearchCV(bst, xgb_params, n_jobs = -1, scoring = 'roc_auc')
# grid_search_xgb.fit(X_tree_train, y_tree_train, sample_weight = classes_weights)
grid_search_xgb.fit(X_tree_train, y_tree_train)


print(grid_search_xgb.best_estimator_)

print(f'The XGB model accuracy on the training set is {grid_search_xgb.score(X_tree_train, y_tree_train):.2f}')

y_xgb_pred = grid_search_xgb.predict(X_tree_test)
print(f'The XGB model accuracy on the test set is {accuracy_score(y_tree_test, y_xgb_pred):.2f}')
y_xgb_pred_proba = grid_search_xgb.predict_proba(X_tree_test)[: , 1]


disp = ConfusionMatrixDisplay(confusion_matrix(y_tree_test, y_xgb_pred))
disp.plot()
disp.ax_.set_title("Confusion matrix for XGB model")
    
    
fpr, tpr, thresholds = roc_curve(y_tree_test, y_xgb_pred_proba)
auc_calc = auc(fpr, tpr)
roc = RocCurveDisplay(fpr = fpr, tpr = tpr, roc_auc = auc_calc)
roc.plot()
roc.ax_.set_title("ROC Curve for XGB model")


#%%
# Get the feature importances
# Baseline model:

initial_logistic_odds_df = pd.DataFrame(np.exp(initial_logistic.coef_[0]), index = X_initial_train.columns, 
                                        columns = ['Odds'])
print(initial_logistic_odds_df.sort_values(by='Odds', ascending = False))    

# Revised logistic 1 (no additional columns removed)




#%%
# Let's try using RFE (recursive feature elimination) - will try 
# # https://machinelearningmastery.com/rfe-feature-selection-in-python/
# def get_models():
#     models = dict()
#     # Let's try this range
#     for i in range(5, 21):
#         rfe = RFE(LogisticRegression(max_iter = 500), n_features_to_select= i)    
#         models[str(i)] = Pipeline([('transform', rfe), ('model', LogisticRegression())])
#     return models
    
# def evaluate_model(name, model, X_train, y_train, X_test, y_test):
#     model.fit(X_train, y_train)
#     print(f"The model's accuracy on the training set w/{name} features is: {model.score(X_train, y_train):.2f}")

#     y_pred = model.predict(X_test)
#     print(f"The model's accuracy on the test set w/{name} features is {accuracy_score(y_test, y_pred):.2f}")

    
# models = get_models()
# for name, model in models.items():
#     evaluate_model(name, model, X_train, y_train, X_test, y_test)


# rfe = RFE(LogisticRegression(max_iter = 500), n_features_to_select= 10)
# pipe = Pipeline([('transform', rfe), ('model', LogisticRegression(max_iter = 500))])
# pipe.fit(X_train, y_train)

#%%
# ABANDONED
# Now going to do some feature selection
# Start with the numerical variables to identify correlated features:
# Example from https://www.analyticsvidhya.com/blog/2020/03/what-is-multicollinearity/

# def calc_vif(numeric_cols_df):
#     # Calculating VIF
#     vif = pd.DataFrame()
#     vif["variables"] = numeric_cols_df.columns
#     vif["VIF"] = [variance_inflation_factor(numeric_cols_df.values, i) for i in range(numeric_cols_df.shape[1])]

#     return vif

# # Exclude this column because of all 0 values...will just drop it in next models as it has no value
# # Also going to use the original df.

# numeric_cols_df = df[numeric_cols].drop(columns = 'device_fraud_count')
# calc_vif(numeric_cols_df)

# # velocity_24h, velocity_6h, velocity_4w, credit_risk_score, and device_distinct_emails_8w 
# # all had VIF values > 5. Will remove the highest (velocity_4w) and rerun
# numeric_cols_df.drop(columns = 'velocity_4w', inplace = True)
# calc_vif(numeric_cols_df)

# # Continue process
# numeric_cols_df.drop(columns = 'device_distinct_emails_8w', inplace = True)
# calc_vif(numeric_cols_df)

# numeric_cols_df.drop(columns = 'velocity_24h', inplace = True)
# calc_vif(numeric_cols_df)

# # Credit risk score remains over 5 (5.46) but I will keep it as I believe it to be a critical predictor
# # So in total, we removed 4 features - device_fraud_count, velocity_4w, device_distinct_emails_8w,
# # and velocity_24h

# # Save the columns that will be used
# selected_numeric_cols = list(numeric_cols_df.columns)

#%%
# ABANDONED
# Now perform feature selection on categorical variables.
# Following tutorial from https://machinelearningmastery.com/feature-selection-with-categorical-data/
# https://towardsdatascience.com/chi-square-test-for-feature-selection-in-machine-learning-206b1f0b8223
# # First need to ordinal encode variables.
# ordinal_encoder2 = OrdinalEncoder()
# # Get all categorical variables
# categorical_cols = df[nominal_cols + [ordinal_col]]

# ordinal_encoder2.fit(categorical_cols)





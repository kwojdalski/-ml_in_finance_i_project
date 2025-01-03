# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# predicting the return of a stock in the US market using historical data over a recent period of 20 days
#
# Agenda:
# 1. **Data Preprocessing**
#    - Loading training and test datasets
#    - Handling missing values
#    - Feature engineering (moving averages)
#    - Target encoding
#
# 2. **Model Implementation and Evaluation**
#    - Decision Tree Classifier
#       - Baseline model (accuracy: 0.510)
#       - Tuned model with hyperparameter optimization (accuracy: 0.5325)
#    - XGBoost Classifier
#       - Baseline model (accuracy: 0.53)
#       - Tuned model with hyperparameter optimization (accuracy: 0.8775)
#
# 3. **Model Comparison**
#    - Cross-validation results
#    - Feature importance analysis
#    - ROC curves and confusion matrices
#
# The goal is to beat the benchmark accuracy of 0.5131 using various modeling approaches and optimization techniques.
# %% [markdown]
# ## Data description
#
# 3 datasets are provided as csv files, split between training inputs and outputs, and test inputs.
#
# Input datasets comprise 47 columns: the first ID column contains unique row identifiers while the other 46 descriptive features correspond to:
#
# * **DATE**: an index of the date (the dates are randomized and anonymized so there is no continuity or link between any dates),
# * **STOCK**: an index of the stock,
# * **INDUSTRY**: an index of the stock industry domain (e.g., aeronautic, IT, oil company),
# * **INDUSTRY_GROUP**: an index of the group industry,
# * **SUB_INDUSTRY**: a lower level index of the industry,
# * **SECTOR**: an index of the work sector,
# * **RET_1 to RET_20**: the historical residual returns among the last 20 days (i.e., RET_1 is the return of the previous day and so on),
# * **VOLUME_1 to VOLUME_20**: the historical relative volume traded among the last 20 days (i.e., VOLUME_1 is the relative volume of the previous day and so on),
#
# Output datasets are only composed of 2 columns:
#
# * **ID**: the unique row identifier (corresponding to the input identifiers)
# and the binary target:
# * **RET**: the sign of the residual stock return at time $t$
#
# The one-day return of a stock :
# $R^t$ =  $\frac{P_j^t}{P_j^{t-1}}$ - 1
#
# The solution files submitted by participants shall follow this output dataset format (i.e contain only two columns, ID and RET, where the ID values correspond to the input test data).
# An example submission file containing random predictions is provided.
#
# **418595 observations (i.e. lines) are available for the training datasets while 198429 observations are used for the test datasets.**
#


# %% [markdown]
# ## Importing libraries


# %%
import logging as log
from itertools import compress

import numpy as np
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, train_test_split

from src.utils import feature_importance, load_and_preprocess_data, model_fit

# %%
# Configure logging to stdout
log.basicConfig(
    level=log.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[log.StreamHandler()],
)
target = "RET"
kfold = 2


# %% [markdown]
# ## Importing data

# %%
train_df, test_df = load_and_preprocess_data()

# Display sample of training data
train_df.head(2)

# %%
# Features Default selection
features = train_df.columns.drop(target).tolist()

# %% [markdown]
# ## ML DecisionTreeClassifier

# %% [markdown]
# Train and test set splitting
x_train, x_test, y_train, y_test = train_test_split(
    train_df.loc[:, train_df.columns != "RET"],
    train_df["RET"],
    test_size=0.25,
    random_state=0,
)

# %%
# Decison tree baseline model
model = tree.DecisionTreeClassifier()

# %%
# Fitting Decison tree baseline model
model_fit(model, x_train, y_train, features, performCV=False)
log.info("Accuracy on test set :{:.3f} ".format(model.score(x_test, y_test)))

# %%
# Tunning Decision tree model  With Gridsearch
log.info("Decision tree with Classifier")
params = {"max_depth": np.arange(2, 7), "criterion": ["gini", "entropy"]}
tree_estimator = tree.DecisionTreeClassifier()

grid_tree = GridSearchCV(
    tree_estimator, params, cv=kfold, scoring="accuracy", n_jobs=1, verbose=False
)

grid_tree.fit(x_train, y_train)
best_est = grid_tree.best_estimator_
log.info(best_est)
log.info(grid_tree.best_score_)


# summarize results
log.info("Best: %f using %s" % (grid_tree.best_score_, grid_tree.best_params_))
means = grid_tree.cv_results_["mean_test_score"]
stds = grid_tree.cv_results_["std_test_score"]
params = grid_tree.cv_results_["params"]
for mean, stdev, param in zip(means, stds, params):
    log.info("%f (%f) with: %r" % (mean, stdev, param))


# %% [markdown]
# **the best Hyperparameters for our Decision tree model using gridsearch Cv  is {'criterion': 'gini', 'max_depth': 6}**

# %%
model = tree.DecisionTreeClassifier(max_depth=6, criterion="gini")
model_fit(model, x_train, y_train, features, printFeatureImportance=True)

dt = tree.DecisionTreeClassifier()
dt.fit(x_train, y_train)

# plot tree
# plt.figure(figsize=(20, 16))  # set plot size (denoted in inches)
# tree.plot_tree(model, feature_names=features, fontsize=10)

# %% [markdown]
# ## features selection based on Feature importances

# %%
feature_importance(model, features)

# %% [markdown]
# **we removed features with less than 2% of feature importance**
n_features = list(compress(features, model.feature_importances_ >= 0.02))

# %%
# New sets with only the selected features
x_train_sl, x_test_sl, y_train_sl, y_test_sl = train_test_split(
    train_df.loc[:, train_df[n_features].columns], train_df["RET"], random_state=0
)

# %%
model_selected = tree.DecisionTreeClassifier(max_depth=6, criterion="gini")

# %%
log.info("Fitting with train set")
model_fit(
    model_selected, x_train_sl, y_train_sl, n_features, printFeatureImportance=True
)

# %%
log.info("Fitting with test set")
model_fit(
    model_selected,
    x_test_sl,
    y_test_sl,
    n_features,
    printFeatureImportance=False,
    roc=True,
)

# %%
# Prediction on the test dataframe
test_df = test_df[n_features]
prediction = model_selected.predict(test_df)
log.info(prediction)

# %% [markdown]
# ## ML GradientBoostingClassifier

# %%
# Train and test set splitting
x_train, x_test, y_train, y_test = train_test_split(
    train_df.loc[:, train_df.columns != "RET"],
    train_df["RET"],
    test_size=0.25,
    random_state=0,
)

# %%
# Baseline Gradient boosting model
base_gbm = GradientBoostingClassifier(random_state=10)
model_fit(base_gbm, x_train, y_train, features, roc=True, printFeatureImportance=True)

# %% [markdown]
# **Tunning parameters with Gridsearch**
# ** Baseline approch**
#    *Fix learning rate and number of estimators for tuning tree-based parameters
#     min_samples_split = 500 : This should be ~0.5-1% of total values.
#     min_samples_leaf = 50 :  for preventing overfitting and again a small value.
#     max_depth = 8 : Should be chosen (5-8) based on the number of observations and predictors.
#     max_features = 'sqrt' : Its a general thumb-rule to start with square root.
#     subsample = 0.8 : commonly used used start value
#
# **we will choose all the features

# %%
log.info("tuning n_estimators")
params1 = {"n_estimators": range(30, 81, 10)}

estimator = GradientBoostingClassifier(
    learning_rate=0.1,
    min_samples_split=500,
    min_samples_leaf=50,
    max_depth=8,
    max_features="sqrt",
    subsample=0.8,
    random_state=10,
)

grid_xgb1 = GridSearchCV(
    estimator, params1, cv=5, scoring="accuracy", n_jobs=1, verbose=False
)
grid_result = grid_xgb1.fit(x_train, y_train)

# %%
log.info("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_["mean_test_score"]
stds = grid_result.cv_results_["std_test_score"]
params = grid_result.cv_results_["params"]
for mean, stdev, param in zip(means, stds, params):
    log.info("%f (%f) with: %r" % (mean, stdev, param))

# %%
log.info("tuning max_depth and min_sample_split")
params2 = {"max_depth": range(5, 16, 2), "min_samples_split": range(400, 1001, 200)}

estimator = GradientBoostingClassifier(
    learning_rate=0.1,
    n_estimators=80,
    max_features="sqrt",
    subsample=0.8,
    random_state=10,
)

grid_xgb2 = GridSearchCV(
    estimator, params2, cv=5, scoring="accuracy", n_jobs=-1, verbose=True
)

grid_result = grid_xgb2.fit(x_train, y_train)

log.info("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_["mean_test_score"]
stds = grid_result.cv_results_["std_test_score"]
params = grid_result.cv_results_["params"]
for mean, stdev, param in zip(means, stds, params):
    log.info("%f (%f) with: %r" % (mean, stdev, param))

# %% [markdown]
# tuning max_depth and min_sample_split
# Fitting 5 folds for each of 24 candidates, totalling 120 fits
# [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
# [Parallel(n_jobs=1)]: Done 120 out of 120 | elapsed: 134.9min finished
# Best: 0.540790 using {'max_depth': 15, 'min_samples_split': 400}
# 0.532043 (0.000315) with: {'max_depth': 5, 'min_samples_split': 400}
# 0.532069 (0.000926) with: {'max_depth': 5, 'min_samples_split': 600}
# 0.532281 (0.000915) with: {'max_depth': 5, 'min_samples_split': 800}
# 0.531996 (0.001207) with: {'max_depth': 5, 'min_samples_split': 1000}
# 0.535426 (0.001870) with: {'max_depth': 7, 'min_samples_split': 400}
# 0.534912 (0.000907) with: {'max_depth': 7, 'min_samples_split': 600}
# 0.535163 (0.001501) with: {'max_depth': 7, 'min_samples_split': 800}
# 0.534989 (0.001050) with: {'max_depth': 7, 'min_samples_split': 1000}
# 0.537268 (0.001484) with: {'max_depth': 9, 'min_samples_split': 400}
# 0.537225 (0.001056) with: {'max_depth': 9, 'min_samples_split': 600}
# 0.536886 (0.002497) with: {'max_depth': 9, 'min_samples_split': 800}
# 0.535931 (0.000559) with: {'max_depth': 9, 'min_samples_split': 1000}
# 0.538961 (0.001785) with: {'max_depth': 11, 'min_samples_split': 400}
# 0.538715 (0.001752) with: {'max_depth': 11, 'min_samples_split': 600}
# 0.539016 (0.002292) with: {'max_depth': 11, 'min_samples_split': 800}
# 0.537764 (0.002626) with: {'max_depth': 11, 'min_samples_split': 1000}
# 0.540009 (0.000849) with: {'max_depth': 13, 'min_samples_split': 400}
# 0.539254 (0.001786) with: {'max_depth': 13, 'min_samples_split': 600}
# 0.538851 (0.001672) with: {'max_depth': 13, 'min_samples_split': 800}
# 0.539967 (0.001591) with: {'max_depth': 13, 'min_samples_split': 1000}
# 0.540790 (0.002216) with: {'max_depth': 15, 'min_samples_split': 400}
# 0.539504 (0.001396) with: {'max_depth': 15, 'min_samples_split': 600}
# 0.540633 (0.002234) with: {'max_depth': 15, 'min_samples_split': 800}
# 0.539700 (0.002217) with: {'max_depth': 15, 'min_samples_split': 1000}

# %%
# the best parameter is give by Best: 0.540790 using {'max_depth': 15, 'min_samples_split': 400}
log.info("tuning num_sample_split and min_sample_split")
params3 = {
    "min_samples_leaf": range(40, 70, 10),
    "min_samples_split": range(400, 1001, 200),
}
estimator = GradientBoostingClassifier(
    learning_rate=0.1,
    n_estimators=80,
    max_depth=15,
    max_features="sqrt",
    subsample=0.8,
    random_state=10,
)
grid_xgb3 = GridSearchCV(
    estimator, params3, cv=5, scoring="accuracy", n_jobs=-1, verbose=True
)
grid_result = grid_xgb3.fit(x_train, y_train)

log.info("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_["mean_test_score"]
stds = grid_result.cv_results_["std_test_score"]
params = grid_result.cv_results_["params"]
for mean, stdev, param in zip(means, stds, params):
    log.info("%f (%f) with: %r" % (mean, stdev, param))

# %%
# Model fitting of the Grid search best estimator
model_fit(grid_xgb3.best_estimator_, x_train, y_train, features, roc=False)

# %%
model_fit(grid_xgb3.best_estimator_, x_test, y_test, features, roc=False)

# %%
grid_xgb3.best_estimator_

# %%
log.info("tuning max_features")
params4 = {"max_features": range(7, 20, 2)}

estimator = GradientBoostingClassifier(
    learning_rate=0.1,
    n_estimators=80,
    max_depth=15,
    min_samples_split=400,
    min_samples_leaf=40,
    subsample=0.8,
    random_state=10,
)
grid_xgb4 = GridSearchCV(
    estimator, params4, cv=5, scoring="accuracy", n_jobs=1, verbose=True
)
grid_result = grid_xgb4.fit(x_train, y_train)

log.info("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_["mean_test_score"]
stds = grid_result.cv_results_["std_test_score"]
params = grid_result.cv_results_["params"]
for mean, stdev, param in zip(means, stds, params):
    log.info("%f (%f) with: %r" % (mean, stdev, param))

# %%
xgb_tunned = GradientBoostingClassifier(
    learning_rate=0.1,
    n_estimators=80,
    max_depth=19,
    min_samples_split=400,
    min_samples_leaf=40,
    subsample=0.8,
    random_state=1,
)

# %%
# Fit Cross validation and prediction on the train and the test set
model_fit(
    xgb_tunned,
    x_train,
    y_train,
    features,
    performCV=True,
    roc=False,
    printFeatureImportance=True,
)
model_fit(xgb_tunned, x_test, y_test, features, performCV=True, roc=False)
model_fit(xgb_tunned, x_test, y_test, features, performCV=True, roc=False)
model_fit(xgb_tunned, x_test, y_test, features, performCV=True, roc=False)

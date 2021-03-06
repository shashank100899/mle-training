import argparse as ar
import logging as lg
import os
import pickle
import urllib.request

import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedShuffleSplit,
    train_test_split,
)
from sklearn.tree import DecisionTreeRegressor

lg.basicConfig(filename="ingest_data_log.txt", level=lg.INFO, format="%(asctime)s %(message)s")
lg.info("The score script stated with logging")

parser = ar.ArgumentParser()
parser.add_argument("models_folder")
parser.add_argument("housing_prepared")
parser.add_argument("strat_test_set")
parser.add_argument("housing_labels")
args = parser.parse_args()

lg.info("loaded all the requied inputs for the Score script")

housing_prepared = pd.read_csv(args.housing_prepared)
housing_labels = pd.read_csv(args.housing_labels).squeeze()
strat_test_set = pd.read_csv(args.strat_test_set)

lin_reg = pickle.load(open(os.path.join(args.models_folder, "linear_regression"), "rb"))

lg.info("loaded the Linear Regression model using the pickle module")

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse

lg.info("Predicted the o/p using Linear Regression module")

lin_mae = mean_absolute_error(housing_labels, housing_predictions)
lin_mae

tree_reg = pickle.load(open(os.path.join(args.models_folder, "decision_tree"), "rb"))

lg.info("loaded the Decision Tree model using pickel module")

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse

param_distribs = {'n_estimators': randint(low=1, high=200), 'max_features': randint(low=1, high=8)}

forest_reg = pickle.load(open(os.path.join(args.models_folder, "ransom_forest"), "rb"))

lg.info("loaded the random forest model using pickel module")

rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(housing_prepared, housing_labels.ravel())
cvres = rnd_search.cv_results_

lg.info("Did the Random Search using Random forest module ")

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


param_grid = [
    # try 12 (3??4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2??3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}]

# train across 5 folds, that's a total of (12+6)*5=90 rounds of training
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepared, housing_labels.ravel())

grid_search.best_params_
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

feature_importances = grid_search.best_estimator_.feature_importances_
sorted(zip(feature_importances, housing_prepared.columns), reverse=True)


final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

imputer = pickle.load(open(os.path.join(args.models_folder, "imputer"), "rb"))

X_test_num = X_test.drop('ocean_proximity', axis=1)
X_test_prepared = imputer.transform(X_test_num)
X_test_prepared = pd.DataFrame(X_test_prepared, columns=X_test_num.columns, index=X_test.index)
X_test_prepared["rooms_per_household"] = X_test_prepared["total_rooms"] / X_test_prepared["households"]
X_test_prepared["bedrooms_per_room"] = X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
X_test_prepared["population_per_household"] = X_test_prepared["population"] / X_test_prepared["households"]
X_test_cat = X_test[['ocean_proximity']]
X_test_prepared = X_test_prepared.join(pd.get_dummies(X_test_cat, drop_first=True))


final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

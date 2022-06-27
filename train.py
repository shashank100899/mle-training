import argparse as ar
import logging as lg
import pickle

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

def logging_msg(s):
    """ This function will take the logging message and load it
        arugument s is the message which been logged"""
    lg.info(s)

def bining(df, c):
    """This function will take a coloumn and create bins based on the given values
    df is the DataFrame and c is the column name"""
    return pd.cut(df["c"], bins=[0., 1.5, 3.0, 4.5, 6., np.inf], labels=[1, 2, 3, 4, 5])


if __name__ == "__main__":

    parser = ar.ArgumentParser()
    parser.add_argument("--file", default="datasets_ingest_data/housing/housing.csv")
    args = parser.parse_args()

    logging_msg("Took the file as the Argument")

    housing = pd.read_csv(args.file)

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    housing["income_cat"] = bining(housing, "median_income")

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    def income_cat_proportions(data):
        """ this the income_cat_proportion function
        """
        return data["income_cat"].value_counts() / len(data)

    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    compare_props = pd.DataFrame({
        "Overall": income_cat_proportions(housing),
        "Stratified": income_cat_proportions(strat_test_set),
        "Random": income_cat_proportions(test_set),
    }).sort_index()

    compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100

    compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    strat_test_set.to_csv("strat_test_set.csv", index=False)

    housing = strat_train_set.copy()
    housing.plot(kind="scatter", x="longitude", y="latitude")
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

    corr_matrix = housing.corr()
    corr_matrix["median_house_value"].sort_values(ascending=False)
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]

    housing = strat_train_set.drop("median_house_value", axis=1)  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()

    housing_labels.to_csv("housing_labels.csv", index=False)

    imputer = SimpleImputer(strategy="median")

    housing_num = housing.drop('ocean_proximity', axis=1)

    imputer.fit(housing_num)

    with open("/home/shashank1/my_project2/mle-training/models/imputer", "wb") as f:
        pickle.dump(imputer, f)

    logging_msg("Saved the Imputer model using Pickel module")

    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
    housing_tr["rooms_per_household"] = housing_tr["total_rooms"] / housing_tr["households"]
    housing_tr["bedrooms_per_room"] = housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    housing_tr["population_per_household"] = housing_tr["population"] / housing_tr["households"]

    housing_cat = housing[['ocean_proximity']]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

    housing_prepared.to_csv("housing_prepared.csv", index=False)

    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)

    with open("/home/shashank1/my_project2/mle-training/models/linear_regression", "wb") as f:
        pickle.dump(lin_reg, f)

    logging_msg("Saved the linear Regression model using Pickel module")

    tree_reg = DecisionTreeRegressor(random_state=42)
    tree_reg.fit(housing_prepared, housing_labels)

    with open("/home/shashank1/my_project2/mle-training/models/decision_tree", "wb") as f:
        pickle.dump(tree_reg, f)

    logging_msg("Saved the Decision Tree Regression model using Pickel module")

    forest_reg = RandomForestRegressor(random_state=42)

    with open("/home/shashank1/my_project2/mle-training/models/ransom_forest", "wb") as f:
        pickle.dump(forest_reg, f)

    logging_msg("Saved the Random Forest Regressor  model using Pickel module")

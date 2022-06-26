# # import os
# # import pickle

# # from sklearn.linear_model import LinearRegression

# # print(os.getcwdb())

# # lin_reg = LinearRegression()
# # with open("/home/shashank1/my_project2/mle-training/models/linear_regression", "wb") as f:
# #     pickle.dump(lin_reg, f)

# # linear = pickle.load(open("/home/shashank1/my_project2/mle-training/models/linear_regression", "rb"))

# import argparse as ar
# import logging as lg
# import pickle

# import numpy as np
# import pandas as pd
# from scipy.stats import randint
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.impute import SimpleImputer
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from sklearn.model_selection import (
#     GridSearchCV,
#     RandomizedSearchCV,
#     StratifiedShuffleSplit,
#     train_test_split,
# )
# from sklearn.tree import DecisionTreeRegressor

# housing = pd.read_csv("datasets/housing/housing.csv")


# housing["income_cat"] = pd.cut(housing["median_income"],
#                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
#                                labels=[1, 2, 3, 4, 5])


# split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# for train_index, test_index in split.split(housing, housing["income_cat"]):
#     strat_train_set = housing.loc[train_index]
#     strat_test_set = housing.loc[test_index]

# print(type(strat_test_set))


import pandas as pd

df = pd.read_csv("housing_labels.csv")
print(df)
df = df.squeeze()
print(df.ravel())

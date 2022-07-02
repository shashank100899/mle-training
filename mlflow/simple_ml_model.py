import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn


def get_data():
    try:
        df = pd.read_csv("/home/shashank1/my_project2/mle-training/mlflow/winequality-red.csv", sep=";")
        print(df.columns)
        print(df.sample(10))
        return df
    except Exception as e:
        raise e

def main():
    df = get_data()
    train, test = train_test_split(df)
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]
    with mlflow.start_run():
        linear = LinearRegression()
        linear.fit(train_x, train_y)
        predicted = linear.predict(test_x)
        rmse, mae, r2 = evaluate(test_y, predicted)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(linear, "Linear Regression")
        print(f"the rmse{rmse}")
        print(f"the MAE {mae}")
        print(f"the r2 value {r2}")

def evaluate(actual, predicted):
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    return rmse, mae, r2

if __name__ == "__main__":
    main()

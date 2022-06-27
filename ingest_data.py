import argparse as ar
import logging as lg
import os
import tarfile
import urllib.request

import pandas as pd


def logging_msg(s):
    """ This function will take the logging message and load it
        arugument s is the message which been logged"""
    lg.info(s)


def fetch_housing_data(housing_url, housing_path):
    """The fetch_housing_data fuction will connect with githud with the given URL
    and fetch the data and will place the data in the folder"""
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    lg.info("Extracted the file from the URL and place the" + args.path)

def load_housing_data(housing_path):
    """ The load_housing_data fuction with extract the tra files
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    lg.info("Stored the data in DataFrame")
    return pd.read_csv(csv_path)


if __name__ == "__main__":
    lg.basicConfig(filename="ingest_data_log.txt", level=lg.INFO, format="%(asctime)s %(message)s")
    logging_msg("The ingest_data script stated with logging")

    parser = ar.ArgumentParser()
    parser.add_argument("--path", default="datasets_ingest_data/housing/")
    args = parser.parse_args()
    print(args.path)

    logging_msg("Took the file path from the input")

    DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    HOUSING_PATH = args.path
    HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

    fetch_housing_data(HOUSING_URL, HOUSING_PATH)
    housing = load_housing_data(HOUSING_PATH)

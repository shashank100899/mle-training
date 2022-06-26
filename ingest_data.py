import argparse as ar
import logging as lg
import os
import tarfile
import urllib.request

import pandas as pd

lg.basicConfig(filename="ingest_data_log.txt", level=lg.INFO, format="%(asctime)s %(message)s")
lg.info("The ingest_data script stated with logging")

parser = ar.ArgumentParser()
parser.add_argument("path")
args = parser.parse_args()
print(args.path)

lg.info("Took the file path from the input")

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = args.path
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    lg.info("Extracted the file from the URL and place the" + args.path)


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    lg.info("Stored the data in DataFrame")
    return pd.read_csv(csv_path)


fetch_housing_data()
housing = load_housing_data()

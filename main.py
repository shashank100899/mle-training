import os

import mlflow
import mlflow.sklearn

if __name__ == "__main__":
    with mlflow.start_run(run_name='PARENT_RUN') as parent_run:
        mlflow.log_param("parent", "yes")
        with mlflow.start_run(run_name='CHILD_RUN', nested=True) as child_run:
            os.system("python3 ingest_data.py")
            os.system("python3 train.py")
            os.system("python3 score.py")

import os
import pandas as pd
import joblib
import mlflow
import yaml
import pickle
import json

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

mlflow.set_tracking_uri("https://dagshub.com/saraniz/Drift-Monitoring-System.mlflow")

# config file load
# safe_load() → a function inside the yaml module
# Reads YAML content from the file
# Converts it into a Python object (usually a dictionary)
# Returns that object so your program can use it
def load_config():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(BASE_DIR, "config.yaml")

    with open(config_path, "r") as file:
        return yaml.safe_load(file)


# training function
def train_on_dataset(dataset_name, config):

    print(f"Training on dataset: {dataset_name}")

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # get target column for the dataset
    target = config["datasets"][dataset_name]["target"]

    # ABSOLUTE PATHS (FIXED)
    train_path = os.path.join(BASE_DIR, "data", "processed", dataset_name, "train.csv")
    test_path = os.path.join(BASE_DIR, "data", "processed", dataset_name, "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # split features and labels
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]

    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    # set mlflow experiment
    mlflow.set_experiment("drift-monitoring-system")

    # start mlflow run
    with mlflow.start_run(run_name=dataset_name):

        n_estimators = 100

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42
        )

        # train the model
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        # calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average="weighted")
        recall = recall_score(y_test, predictions, average="weighted")
        f1 = f1_score(y_test, predictions, average="weighted")

        # log parameters
        mlflow.log_param("dataset", dataset_name)
        mlflow.log_param("n_estimators", n_estimators)

        # log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # MODEL SAVE (ABSOLUTE PATH FIXED)
        model_dir = os.path.join(BASE_DIR, "models")
        os.makedirs(model_dir, exist_ok=True)

        model_path = os.path.join(model_dir, "model.pkl")

        joblib.dump(model, model_path)

        mlflow.log_artifact(model_path)

        print(f"Finished training {dataset_name}")
        print(f"Accuracy: {accuracy}")


def load_drift_report():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(BASE_DIR, "reports", "drift_report.json")

    with open(path, "r") as f:
        return json.load(f)


def choose_best_dataset(drift_report):
    best_dataset = None
    best_score = float("inf")

    for dataset, values in drift_report.items():
        if values["drift_score"] < best_score:
            best_score = values["drift_score"]
            best_dataset = dataset

    return best_dataset


def main():

    config = load_config()

    # load drift report
    drift_report = load_drift_report()

    # choose best dataset automatically
    best_dataset = choose_best_dataset(drift_report)

    print(f"\n Selected dataset for training: {best_dataset}")

    # train only best dataset
    train_on_dataset(best_dataset, config)


if __name__ == "__main__":
    main()
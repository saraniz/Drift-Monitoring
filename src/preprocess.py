# Preprocessing script for multiple datasets using config.yaml

import os
import pandas as pd
import yaml

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# with ensure autoclosing the file and 'r' mean read only
def load_config():
    """
    Loads config.yaml using absolute path based on script location
    (safe for DVC, MLflow, CLI, pipelines)
    """

    config_path = os.path.join(PROJECT_ROOT, "config.yaml")

    with open(config_path, "r") as file:
        return yaml.safe_load(file)


#create and return preprocessing pipeline based on categorical and numerical features
def build_preprocessor(categorical_features, numerical_features):
    """
    Handles:
    - Missing values
    - Encoding categorical variables
    """

    # Numerical pipeline: fill missing values with median
    #Pipeline is built in library in sklearn come from sklearn.pipeline which chain multiple preprocessing steps in order
    # step= [] return list of tuples. each tuple format ("step_name", transformer_object)
    # "imputer" → name of this step
    # SimpleImputer(...) → fills missing values [built in function]
    # strategy="median" → replaces missing values with median of column
    num_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    # Categorical pipeline:
    # fill missing + one-hot encode
    # handle_unknown="ignore"
    # if new unseen category appears in test data → ignore instead of crashing
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine both pipelines
    # ColumnTransformer(...) Used to apply different pipelines to different column groups.Applies different transformations to different column groups.
    # transformers is list oft tuples
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numerical_features),
            ("cat", cat_pipeline, categorical_features)
        ]
    )

    return preprocessor


# load the dataset from config and do the preprocessing...name mean dataset name
def preprocess_dataset(name, config):
    """
    Processes a single dataset using config rules
    """

    print(f"\nProcessing dataset: {name}")

    # config → dictionary loaded from YAML
    # "datasets" → top-level key
    # [name] → specific dataset section (e.g., "churn")
    ds_config = config["datasets"][name]

    # Resolve relative paths from project root so script works from any cwd
    data_path = ds_config["file_path"]
    if not os.path.isabs(data_path):
        data_path = os.path.join(PROJECT_ROOT, data_path)

    # Load dataset from config path
    df = pd.read_csv(data_path)

    print("Original shape:", df.shape)

    target = ds_config["target"]
    categorical_features = ds_config["categorical_features"]
    numerical_features = ds_config["numerical_features"]

    # Split into X and y
    X = df.drop(columns=[target])
    y = df[target]

    # Build preprocessing pipeline
    preprocessor = build_preprocessor(categorical_features, numerical_features)

    # Fit + transform
    # .fit() learns from data (mean, categories, etc.)
    # .transform() applies learned transformation
    # .fit_transform() does both in one step
    # Result: returns processed numeric matrix (often NumPy array or sparse matrix)
    X_processed = preprocessor.fit_transform(X)

    # Get feature names
    # built-in sklearn method (new versions)
    # Purpose: Gets names of transformed columns like:
    # num__Age
    # cat__Gender_Male
    # cat__Gender_Female
    feature_names = preprocessor.get_feature_names_out()

    # Convert back to DataFrame
    # hasattr checks if object has attribute (like toarray for sparse matrix)
    X_df = pd.DataFrame(
        X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed,
        columns=feature_names
    )

    # Combine features + target
    # axis=1 → join columns side by side
    # reset_index(drop=True) → aligns indexes
    processed_df = pd.concat([X_df, y.reset_index(drop=True)], axis=1)

    # Split data
    train_df, test_df = train_test_split(
        processed_df,
        test_size=0.2,
        random_state=42
    )

    # ABSOLUTE PATH FIX STARTS HERE
    output_dir = os.path.join(PROJECT_ROOT, "data", "processed", name)

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Saved: {output_dir}")
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)


def main():

    # Load config file
    config = load_config()

    # Loop through all datasets defined in YAML
    for dataset_name in config["datasets"]:
        preprocess_dataset(dataset_name, config)


# Entry point
if __name__ == "__main__":
    main()
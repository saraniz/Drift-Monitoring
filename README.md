# Drift Monitoring System

This project builds a simple end-to-end MLOps workflow for churn modeling with data drift simulation, drift detection, and automated training.

## What I implemented

1. Added two drift generation scripts:
	- driftscripts/script1.py for mild drift
	- driftscripts/script2.py for stronger drift
2. Built a preprocessing pipeline in src/preprocess.py:
	- Loads dataset rules from config.yaml
	- Handles missing values
	- One-hot encodes categorical features
	- Splits train and test sets
	- Saves processed outputs in data/processed/
3. Built drift detection in src/drift.py:
	- Compares baseline churn vs drifted datasets
	- Computes mean difference, variance difference, and KS statistic
	- Produces reports/drift_report.json for DVC and training decisions
4. Built model training in src/train.py:
	- Reads drift report
	- Selects lowest drift dataset automatically
	- Trains RandomForestClassifier
	- Logs params, metrics, and model artifacts to MLflow (DagsHub)
5. Added DVC pipeline stages in dvc.yaml:
	- preprocess
	- drift
	- train
6. Added .gitignore with data-friendly rules:
	- Ignores CSV files
	- Keeps .csv.dvc files tracked

## Important fixes made

Path handling was fixed so scripts work regardless of where commands are executed from.

- src/preprocess.py now resolves config and dataset paths from the project root.
- src/drift.py now writes reports/drift_report.json using a project-root absolute path.
- driftscripts/script1.py and driftscripts/script2.py now read and write CSV files using script-based absolute paths.

These fixes prevent FileNotFoundError issues when running from different folders.

## Project structure

- config.yaml: dataset and feature configuration
- data/: raw, drifted, and processed datasets
- driftscripts/: drift simulation scripts
- src/preprocess.py: preprocessing pipeline
- src/drift.py: drift detection and report generation
- src/train.py: model training and MLflow logging
- reports/drift_report.json: drift results for pipeline decisions
- models/model.pkl: trained model output
- dvc.yaml: reproducible pipeline definition

## How to run

Run commands from the repository root (folder containing dvc.yaml).

1. Generate drifted datasets:

	python driftscripts/script1.py
	python driftscripts/script2.py

2. Reproduce full DVC pipeline:

	dvc repro

3. Push DVC artifacts:

	dvc push

## Current behavior

- Preprocess stage generates train and test files under data/processed for:
  - churn
  - churn_drifted_v1
  - churn_drifted_v2
- Drift stage classifies both drifted datasets as Strong Drift.
- Train stage selects the lowest drift score dataset and logs run details to MLflow.

Example observed result:
- churn_drifted_v2 accuracy around 0.7193

## Notes

- If dvc repro is run inside src/, it fails because dvc.yaml is at repository root.
- Always run dvc commands from project root.
# Credit Risk MLOps Pipeline

An end-to-end MLOps pipeline for credit risk prediction, built on Databricks. This project demonstrates a production-grade machine learning workflow using the Medallion Architecture, from raw data ingestion to model serving and monitoring.

## Architecture
```
Raw Data Generation
       ↓
  01_bronze.ipynb        # Data ingestion → bronze_credit_raw (Delta Table)
       ↓
  02_silver.ipynb        # Data cleaning → silver_credit (Delta Table)
       ↓
  03_gold.ipynb          # Feature engineering → gold_credit_features (Delta Table)
       ↓
  04_train.ipynb         # Model training + MLflow tracking + Model Registry
       ↓
  05_serve.ipynb         # REST API endpoint validation
       ↓
  06_batch_inference.ipynb  # Batch predictions → gold_predictions (Delta Table)
       ↓
  07_monitoring.ipynb    # Data drift detection → monitoring_drift_log (Delta Table)
```

## Tech Stack

- **Platform**: Databricks (Serverless)
- **Data Format**: Delta Lake
- **Processing**: Apache Spark (PySpark)
- **ML Tracking**: MLflow
- **Model Registry**: Unity Catalog
- **Model Serving**: Databricks Model Serving (REST API)
- **Orchestration**: Databricks Workflows
- **Language**: Python

## Pipeline Overview

### 01 Bronze — Data Ingestion
- Generates synthetic credit risk dataset (100,000 records) with intentionally injected dirty data
- Stores raw data as a Delta Table: `bronze_credit_raw`
- Injected issues: negative age values, invalid credit scores (9999), null income and loan amounts

### 02 Silver — Data Cleaning
- Removes records with invalid age (`age <= 0`) and negative debt ratio
- Replaces invalid credit scores (> 850) with null, then imputes using median
- Drops records with null values in core fields (`income`, `loan_amount`)
- Output: `silver_credit` (96,000 records)

### 03 Gold — Feature Engineering
- Computes derived features with business meaning:
  - `loan_to_income_ratio`: loan amount relative to annual income
  - `expense_to_income_ratio`: annual expenses relative to income
  - `savings_to_loan_ratio`: savings buffer against loan amount
  - `financial_stress_score`: weighted composite of debt ratio, credit utilization, and late payments
- Output: `gold_credit_features`

### 04 Train — Model Training
- Reads from `gold_credit_features`
- Splits data 80/20 with stratification to preserve class balance
- Trains a `GradientBoostingClassifier` wrapped in a `sklearn Pipeline` with `StandardScaler`
- Logs parameters, metrics (AUC), and model artifact to MLflow
- Registers model to Unity Catalog Model Registry with signature and input example

### 05 Serve — REST API Validation
- Deploys model via Databricks Model Serving endpoint
- Validates end-to-end inference via HTTP POST request
- Returns real-time default prediction for a single applicant

### 06 Batch Inference — Batch Predictions
- Loads registered model using `mlflow.pyfunc.spark_udf` for distributed inference
- Runs predictions across all records in `gold_credit_features`
- Stores results in `gold_predictions` Delta Table with prediction date

### 07 Monitoring — Data Drift Detection
- Compares feature distributions between training data (baseline) and new incoming data
- Flags features with mean shift > 10% as drifted
- Appends daily drift report to `monitoring_drift_log` Delta Table for historical tracking

## Workflow Orchestration

All notebooks are orchestrated using **Databricks Workflows** with sequential task dependencies:
```
bronze → silver → gold → train → batch_inference → monitoring
```

The pipeline can be scheduled to run automatically (e.g., daily at 2 AM) to retrain the model on fresh data and detect drift over time.

## Key Concepts Demonstrated

- **Medallion Architecture**: Bronze / Silver / Gold layered data organization
- **Data Quality**: Null handling, outlier detection, schema validation
- **Feature Engineering**: Business-driven derived features
- **MLflow Integration**: Experiment tracking, model versioning, artifact logging
- **Unity Catalog**: Centralized model governance and access control
- **Distributed Inference**: `spark_udf` for scalable batch prediction
- **Data Drift Monitoring**: Statistical comparison of feature distributions
- **Pipeline Automation**: End-to-end orchestration with dependency management

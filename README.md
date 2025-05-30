Credit Fraud Detector
Overview
This project implements an end-to-end Credit Fraud Detection system using a real-world Kaggle dataset, focused on scalable data ingestion, feature engineering, machine learning modeling, and insightful visualizations.

It is scoped to use:

Python for ETL, modeling, and visualizations.

PostgreSQL for data storage and querying.

Matplotlib (and optionally Seaborn) for visualization.

scikit-learn and imbalanced-learn for modeling pipelines and sampling.

Business Case
Credit card fraud results in significant financial loss globally due to chargebacks and penalties. A high-performing fraud detection model can:

Minimize fraud-related losses.

Reduce false positives to avoid blocking legitimate transactions.

Help financial institutions save operational costs and improve customer trust.

Project Architecture
scss
Copy
Edit
CSV (Kaggle creditcard.csv)
       ↓
Python ETL Script
  - Clean data (IQR outlier removal)
  - Feature engineering (rolling-window features, time dummies)
       ↓
PostgreSQL Database
  - raw_transactions (ingested cleaned data)
  - features_transactions (engineered features)
       ↓
Python Modeling Pipeline (Jupyter Notebook)
  - Data extraction from Postgres
  - Pipeline: Scaler + Sampler + Classifier
  - TimeSeriesSplit CV to avoid data leakage
  - Hyperparameter tuning with GridSearchCV
  - Evaluation metrics & visualizations (ROC, PR curves, confusion matrix, etc.)
Data & ETL
Source: Kaggle creditcard.csv dataset.

Cleaning:

Outliers removed using IQR method on Amount and top-correlated V features.

Feature Engineering:

Rolling-window features capturing recent transaction behavior (count, sum per card in last hour).

Time-of-day and day-of-week dummy variables to capture temporal patterns.

Storage:

Two tables in PostgreSQL:

raw_transactions — cleaned base data.

features_transactions — engineered features for modeling.

Modeling Pipeline
Pipeline steps:

Scaler (StandardScaler or RobustScaler)

Sampler (SMOTE or NearMiss for class balancing)

Classifier (RandomForestClassifier or XGBClassifier)

Validation:

TimeSeriesSplit for time-based cross-validation to prevent leakage.

Hyperparameter tuning:

GridSearchCV or RandomizedSearchCV over parameters such as number of trees, sampling ratio, etc.

Metrics:

ROC-AUC

Precision-Recall AUC

Precision at K

Visualizations
Included in the Jupyter notebook using Matplotlib (+ Seaborn optionally):

Class imbalance bar chart

Correlation heatmap of top-10 V-features vs. Class

ROC and Precision–Recall curves comparing best two models

Feature importance bar chart from RandomForest or XGBoost

Annotated confusion matrix on hold-out test set

(Optional) t-SNE 2D scatter plot of final feature set for visual cluster separation

Deliverables
Python ETL script: Loads CSV → Cleans → Loads to Postgres tables.

Jupyter Notebook: Includes modeling pipeline, CV, metrics, and visualizations.

SQL Schema: PostgreSQL table definitions for raw_transactions and features_transactions.

Sample SQL Queries: Demonstrate fetching new data for model retraining.

Results Summary: Table comparing ROC-AUC, PR-AUC, and precision@k for RandomForest and XGBoost models.

Slide Deck (8–10 slides):

Business Case & problem context.

Architecture overview.

Key visualizations.

Next steps & future enhancements (real-time scoring, drift monitoring).

How to Run
Setup PostgreSQL
Create the database and tables using the provided SQL schema (schema.sql).

Update connection parameters in the ETL script (etl.py).

Run ETL
bash
Copy
Edit
python etl.py
Modeling & Evaluation
Open the Jupyter notebook (credit_fraud_modeling.ipynb) and run all cells to train models, perform CV, and generate visualizations.

Next Steps
Deploy the model for real-time scoring to detect fraud at transaction time.

Implement drift monitoring to maintain model accuracy over time.

Explore additional feature engineering using external data (e.g., merchant info, geographic data).

Automate ETL and retraining pipeline with workflow tools like Airflow.

References
Kaggle Credit Card Fraud Dataset: https://www.kaggle.com/mlg-ulb/creditcardfraud

scikit-learn Documentation: https://scikit-learn.org/

imbalanced-learn Documentation: https://imbalanced-learn.org/

PostgreSQL Documentation: https://www.postgresql.org/docs/


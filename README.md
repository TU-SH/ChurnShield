# ChurnShield

## End-to-End Customer Churn Prediction System Built for the Australian Telco Market

A production-grade machine learning system that predicts which Australian telco customers are at risk of churning, explains why using SHAP, serves real-time predictions through a REST API, tracks every experiment in MLflow, and surfaces live business insights on a Streamlit dashboard — all containerised with Docker. 

Churn costs Australian telcos an estimated $400–$600 per lost customer in re-acquisition costs. Identifying at-risk customers 30 days before they leave — and understanding why they are leaving — lets retention teams intervene with targeted offers before it is too late.

## Australian Market Context
ChurnShield is built specifically for the Australian telco market: 

- State segmentation across all 8 Australian states and territories (NSW · VIC · QLD · WA · SA · TAS · ACT · NT)
- AUD pricing calibrated to real Australian telco charge rates (day: $0.17/min · evening: $0.085/min · night: $0.045/min · international: $0.27/min)
- 14% churn rate matching published benchmarks for Telstra, Optus, and Vodafone AU
- Customer service call patterns calibrated to Australian consumer behaviour (ACCC complaint data)
- The modelling approach — XGBoost with SHAP, threshold tuning, cohort analysis mirrors what data science teams at Telstra, Optus, Afterpay, and Airwallex use in production

## Tech Stack

| Layer           |Technology          |Version                | Purpose              | 
|------------|----------|----------------|---------------------|
|ML Model            |XGBoost          |2.0.3                |Gradient boosted trees - best for tabular churn data| 
|Explainability            |SHAP          |0.45.0                |Per-prediction feature attribution| 
|Calibration            |Scikit-Learn          |1.4.2                |Platt scaling for probability calibration| 
|Experiment Tracking            |MLFlow          |2.11.1                |Params, metrics, model registry| 
|API framework            |FastAPI          |0.111.0                |Async REST API with automatic Swagger docs| 
|Server            |Uvicorn          |0.29.0                |ASGI server for FastAPI| 
|Data validation            |Pydantic v2          |2.7.1                |Request/response schema validation| 
|Dashboard            |Streamlit          |1.35.0                |4-page interactive web app| 
|Charts            |Plotly          |5.22.0          |Interactive charts in dashboard      | 
|Database ORM| SQLAlchemy| 2.0.30| Database engine and session management            |    
|DB driver | psycopg2-binary | 2.9.9 | PostgreSQL connection                          | 
| Data processing | pandas | 2.2.2 | Feature engineering           |  
| Numerics | numpy | 1.26.4 | Array operations           |  
| Containerisation | Docker Compose |  - | 4-service stack           |       
| Language | Python | 3.11           |   -        | 

## System Architecture 

ChurnShield has five layers that talk to each other in one direction:
- Data comes in from a CSV file or Kaggle dataset. A Python script cleans and maps it into the right format, then writes it to PostgreSQL. PostgreSQL holds three schemas: raw customer records, engineered features, and a predictions audit log.
- From PostgreSQL, two things happen in parallel: The training pipeline reads the data, builds the model, and saves it as a `.pkl` file - every run is tracked in MLflow with its parameters, CV AUC score, and SHAP importance. The FastAPI server loads that `.pkl` file on startup and serves predictions through REST endpoints. Every prediction it makes gets logged back to PostgreSQL with the full SHAP breakdown stored as JSONB.
- The Streamlit dashboard sits on top — it reads from PostgreSQL for the Overview, Cohort, and Explain pages, and calls the FastAPI directly for the Live Predict page.

<img width="538" height="649" alt="image" src="https://github.com/user-attachments/assets/d807d8f1-59d5-48e0-bb18-5ca1b481e637" />

## How a prediction works - step by step

A customer record hits POST /predict and six things happen in under 50 milliseconds:

- **Validation**:  Pydantic checks all 19 fields. State must be a valid Australian state code. Numbers must be non-negative. Anything invalid gets rejected immediately with HTTP 422 — the model never sees bad data.

- **Feature engineering**: the 19 raw fields get transformed into 26 features. Key ones: `cs_call_ratio` (customer service calls divided by account age — the strongest churn signal), `total_charge_aud` (sum of all usage charges), `state_encoded` (NSW=0, VIC=1, etc.), and `high_day_usage` (a flag if day minutes exceed the 75th percentile of training data).
  
- **XGBoost + calibration**: the feature vector goes into the trained XGBoost model, which outputs a raw score. Platt scaling converts that score into a true probability between 0 and 1. 
- **SHAP**: the SHAP TreeExplainer computes how much each feature pushed the probability up or down for this specific customer. A `high cs_call_ratio` might add +0.23; a long tenure might subtract -0.09.
-**Risk segment**: the probability is bucketed: below 0.45 is LOW, 0.45–0.70 is MEDIUM, above 0.70 is HIGH. These thresholds were tuned for the real AU telco churn rate of ~14%.
- Two things happen simultaneously: the response goes back to the caller with the probability, risk segment, and top 5 SHAP factors. At the same time, the full prediction including all SHAP values is written to `ml.predictions` as a permanent audit record.
  
<img width="538" height="649" alt="image" src="https://github.com/user-attachments/assets/b0c88088-27d8-429f-a4be-08bf9033da4c" />


## Training Pipeline - step by step 

Training runs with one command — `python -m src.models.train` — and takes about 60 seconds: 

- **Load data**: reads from `data/raw/customers.csv` or PostgreSQL. Prints the row count and churn rate so you can immediately confirm the data loaded correctly.
- **Feature engineering**: runs the same 26-feature pipeline used at inference time. The 75th percentile of `day_mins` is computed here from the training data and saved into the artefact — this prevents data leakage at inference time.
- **Class imbalance**: with ~14% churn, positives are outnumbered ~6:1. XGBoost's `scale_pos_weight` parameter is set to this ratio, which makes the model treat each churner as if they appeared six times. Without this, the model would just predict "not churning" for everyone and still get 86% accuracy.
- **5-fold cross-validation**: the dataset is split into 5 stratified folds, the model is trained on 4 and evaluated on 1, rotated five times. The mean and standard deviation of AUC-ROC across all five folds is logged to MLflow. This is the honest estimate of real-world performance - not the training score.
- **XGBoost model**: trained on the full dataset using the same hyperparameters (400 trees, depth 5, learning rate 0.05, 80% subsampling). Training on all the data after CV is standard practice — the CV score already told us how it generalises.
- **Platt calibration**: `CalibratedClassifierCV` with sigmoid scaling wraps the trained model. This ensures that when it outputs 0.72, it really means "72% of customers with this profile churn" — without calibration, raw XGBoost scores are not true probabilities.
- **SHAP**: TreeExplainer computes SHAP values for every row in the training set. The mean absolute SHAP per feature is calculated and logged to MLflow as a JSON artefact. The top 5 features are printed to the terminal so you can immediately see what the model learned.
- **Save artefact**: everything needed for inference is bundled into one `.pkl` file: the calibrated model, the raw XGBoost model, the SHAP explainer, the feature column names in order, the training-set p75 day minutes, and the churn threshold. The FastAPI server loads this single file at startup and holds it in memory for the lifetime of the process.

<img width="538" height="649" alt="image" src="https://github.com/user-attachments/assets/143ce136-7233-4019-8ff3-e00086e15e26" />


## Model performance
| Metric                | Value                 | Notes              | 
|-----------------|------------------|--------------|
| CV AUC-ROC (5-fold stratified)                | 0.924 ± 0.011                 |Primary metric              | 
| Churn threshold                |0.45                  |Tuned for AU telco              | 
| High-risk threshold                | 0.70                 | Triggers retention action              | 
|API latency                 |< 50ms                  | Single prediction             |
| Batch throughput                | 500 customers per request                 |  -            |
| Test suite                | 27 passed                  | 0 failures             |
|Batch throughput | 500 customers per request | — | 


## SHAP Feature Importance 
Top 5 features ranked by mean absolute SHAP value: 

| Rank                | Feature                 | What it Captures              | Direction         | 
|-----------------|------------------|--------------|---------------|
|1    |  `cs_call_ratio`  | CS calls per day of account life  | ↑ High = churner |
|2     |  `customer_service_calls`  | Raw CS call count   | ↑ High = churner  |
|3    | `international_plan`  | Has international add-on | ↑ Reduces churn risk
|4     | `account_length_days`  | Customer tenure | ↑ Longer = less likely to churn| 
|5      | `voicemail_plan`   | Has voicemail plan  | ↓ Reduces churn risk |

**Sample API Response** 

<img width="385" height="265" alt="image" src="https://github.com/user-attachments/assets/aac19be3-e8a1-4b3a-b4e9-771c5efa3782" />

**Reading the SHAP values**:

- `customer_service_calls`: +0.2341 → this customer's 5 CS calls pushed churn probability up by 23.4 percentage points
- `account_length_days`: -0.0912 → their 3-year tenure pushed it down by 9.1 percentage points
  
Every prediction is explainable, auditable, and logged to PostgreSQL












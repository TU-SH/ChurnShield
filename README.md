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
<img width="538" height="649" alt="image" src="https://github.com/user-attachments/assets/f87da455-e517-4e9b-9a51-4985e984433d" />


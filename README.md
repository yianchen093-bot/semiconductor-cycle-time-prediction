# Semiconductor Cycle Time Prediction

## Project Overview

This project applies machine learning techniques to predict semiconductor manufacturing cycle time and identify important factors affecting production performance.

## Business Problem

Cycle Time is a critical KPI in semiconductor manufacturing because it impacts delivery performance and production efficiency.

## Dataset

15 features:

- lot_id
- product_type
- process_step
- machine_id
- priority
- arrival_time
- cycle_time
- queue_time
- wip_level
- utilization
- downtime_min
- rework_flag
- bottleneck_flag

## Machine Learning Models

- Decision Tree
- Random Forest
- XGBoost

## Results

Random Forest achieved the best prediction performance.

## Technologies

- Python
- Pandas
- Scikit-learn
- Matplotlib

## Author

Yi An Chen

## Regression Results

<img width="4172" height="1481" alt="regression_results" src="https://github.com/user-attachments/assets/9dc7db7a-022c-47e4-adef-40316797793a" />

## Classification Results

<img width="1000" height="500" alt="Classification results" src="https://github.com/user-attachments/assets/4df14a82-e3ac-4bb1-b7a6-a417c0f9ef06" />

## Feature Importance

<img width="4169" height="1482" alt="feature_importance" src="https://github.com/user-attachments/assets/6c189841-1a0e-431f-94fe-f504ce2557c0" />

## Confusion Matrix

<img width="1701" height="1482" alt="confusion_matrix" src="https://github.com/user-attachments/assets/0c185c43-938f-4744-9fc8-84990696f865" />

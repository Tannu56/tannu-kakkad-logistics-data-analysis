# Week 4 - Predictive Modeling and Optimization in Logistics

**Submitted by:** Tannu Kakkad  
**Internship:** Logistics Data Analyst  
**Task:** Week 4 - Predictive Modeling and Optimization in Logistics Systems

## Objective
Forecast `delivery_time_days` using simulated logistics data and translate predictions into operational optimization recommendations.

## Files
- `Week_4_Predictive_Modeling_and_Optimization_Report.docx` - detailed report
- `week4_predictive_model.py` - model training, validation, evaluation and scenario analysis
- `logistics_week4_dataset.csv` - simulated dataset

## Dataset
Features: distance, shipment volume, transportation cost, traffic level, vehicle type and weather condition. Target: delivery time in days.

## Models and Evaluation
Linear Regression and Random Forest Regression are compared using MAE, RMSE and R-squared. Five-fold cross-validation is used for validation.

## Results
Best test-set model by RMSE: **Linear Regression**
- MAE: 1.461
- RMSE: 1.794
- R-squared: 0.682
- Mean 5-fold CV RMSE: 1.954
- Mean 5-fold CV MAE: 1.578

## Optimization
A scenario analysis simulates better dispatch timing or alternative routing by reducing traffic exposure by one level. The simulated average predicted improvement is **1.04 days**. This is an illustrative planning estimate, not a guaranteed saving.

## Run
`pip install pandas numpy scikit-learn`
`python week4_predictive_model.py`

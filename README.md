# Web-Traffic-Forecasting
ML Time Series Regression for daily web traffic forecasting. Uses XGBoost, feature selection (Top 15), and time-series lags. Deployed via Streamlit. Critical Input: Promo Flag (binary 0/1 indicator for marketing sales/events).

This repository contains a full **Time Series Regression** project designed to forecast daily web traffic (`y`) using classical Machine Learning models (XGBoost, Random Forest, Linear Regression) and external marketing features. The final solution is deployed as an interactive prediction dashboard built with **Streamlit**.

## ✨ Project Overview

This project follows a structured pipeline:
1.  **Data Cleaning:** Initial data preparation and exploration (`01_...ipynb`).
2.  **Feature Engineering:** Creation of time-series features (lags, rolling means) (`02_...ipynb`).
3.  **Modeling & Optimization:** Training, feature selection, and saving the best models (`03_...ipynb`).
4.  **Deployment:** Live forecasting application (`streamlit_app.py`).

# 🏦 Bank Customer Churn Predictor

## ### [🔗 View the Live Interactive App Here](https://customerchurn-ah7b2rlbqqrwpujqqh4mna.streamlit.app/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customerchurn-ah7b2rlbqqrwpujqqh4mna.streamlit.app/)

## Project Overview
Customer retention is a critical metric for any financial institution. The cost of acquiring a new banking customer is significantly higher than retaining an existing one. This project provides a machine learning solution to identify customers at high risk of exiting the bank, allowing customer success and retention teams to proactively intervene.

This repository contains a deployed Streamlit web application that utilizes a pre-trained XGBoost model to predict customer churn based on key demographic and financial indicators.

## 🎯 Business Value
By integrating predictive analytics into customer management operations, this tool enables banks to:
* **Reduce Churn Rates:** Identify at-risk profiles before the customer closes their account.
* **Optimize Resource Allocation:** Direct retention budgets and promotional offers specifically toward high-risk, high-value customers rather than a scattergun approach.
* **Improve Operational Efficiency:** Empower relationship managers with quick, data-driven risk assessments during customer interactions.

## 🛠️ Features
* **Interactive UI:** A clean, user-friendly interface built with Streamlit.
* **Real-Time Predictions:** Instant churn risk assessment based on 10 user-input features (e.g., Credit Score, Balance, Tenure, Number of Products).
* **Automated Data Processing:** The app handles backend data formatting, one-hot encoding for categorical variables (Geography, Gender), and feature scaling to match the model's exact training environment.

## 💻 Tech Stack
* **Frontend:** Streamlit
* **Data Manipulation:** Pandas,NumPy
* **Machine Learning:** Scikit-Learn,XGBoost
* **Deployment:** Streamlit Community Cloud

## 📂 Repository Structure

* **app.py:** The main Python script containing the Streamlit application and UI logic.
* **requirements.txt:** The list of Python dependencies required to run the app.
* **churn_model_xgb.pkl:** The serialized, pre-trained XGBoost machine learning model.
* **churn_scaler.pkl:** The fitted scaler used to normalize input data before prediction.





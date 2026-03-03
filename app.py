import streamlit as st
import pandas as pd
import pickle

# --- 1. CONFIGURATION & ASSET LOADING ---
st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

@st.cache_resource
def load_assets():
    # Load the trained model using a relative path
    with open('churn_model_xgb.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
        
    # Load the fitted scaler using a relative path
    with open('churn_scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
        
    return model, scaler

model, scaler = load_assets()

# --- 2. DEFINE EXPECTED TRAINING COLUMNS ---
EXPECTED_COLUMNS = [
    'CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 
    'HasCrCard', 'IsActiveMember', 'EstimatedSalary',
    'Geography_France', 'Geography_Germany', 'Geography_Spain', 
    'Gender_Female', 'Gender_Male'                              
]

# --- 3. STREAMLIT UI & USER INPUTS ---
st.title("🏦 Customer Churn Predictor")
st.write("Enter the customer's details below to predict whether they will exit the bank.")

col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
    geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=5)

with col2:
    balance = st.number_input("Account Balance", min_value=0.0, value=0.0, format="%.2f")
    num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
    has_cr_card = st.selectbox("Has Credit Card?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    is_active_member = st.selectbox("Is Active Member?", options=[1, 0], format_func=lambda x: "Yes" if x == 1 else "No")
    estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, format="%.2f")

# --- 4. PREDICTION LOGIC ---
if st.button("Predict Churn", type="primary"):
    
    # Step A: Convert inputs to a Pandas DataFrame
    input_dict = {
        'CreditScore': [credit_score],
        'Geography': [geography],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    }
    input_df = pd.DataFrame(input_dict)
    
    # Step B: Apply pd.get_dummies
    input_df_encoded = pd.get_dummies(input_df, columns=['Geography', 'Gender'],dtype=int)
    
    # Step C: Align the new dataframe with the expected training columns
    input_df_encoded = input_df_encoded.reindex(columns=EXPECTED_COLUMNS, fill_value=0)
    
    # Step D: Scale the aligned dataframe using the loaded scaler
    # Note: Ensure the scaler was fitted on a dataframe with these exact EXPECTED_COLUMNS
    input_scaled = scaler.transform(input_df_encoded)
    
    # Step E: Make the prediction
    prediction = model.predict(input_scaled)
    
    # Step F: Display results
    st.divider()
    if prediction[0] == 1:
        st.error("⚠️ **High Risk of Churn:** The model predicts this customer will EXIT.")
    else:
        st.success("✅ **Low Risk of Churn:** The model predicts this customer will STAY.")
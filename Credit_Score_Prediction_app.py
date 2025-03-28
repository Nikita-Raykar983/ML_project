import streamlit as st
import pickle
import pandas as pd

# Load the trained model
with open("CreditScore_Model.pkl", "rb") as file:
    model = pickle.load(file)

# Get the feature names the model expects
expected_features = model.feature_names_in_
st.write("Model expects these features:", expected_features)

# Define the Streamlit app
st.title("Credit Score Prediction App")

# Create input fields
month = st.selectbox("Month", list(range(1, 13)))
age = st.number_input("Age", min_value=18, max_value=100, step=1)
occupation = st.selectbox("Occupation", ["Salaried", "Self-Employed", "Business", "Unemployed"])
annual_income = st.number_input("Annual Income", min_value=0.0, step=1000.0)
monthly_inhand_salary = st.number_input("Monthly Inhand Salary", min_value=0.0, step=500.0)
credit_history_age = st.number_input("Credit History Age (in years)", min_value=0.0, step=0.5)
total_emi_per_month = st.number_input("Total EMI per Month", min_value=0.0, step=100.0)
num_bank_accounts = st.number_input("Number of Bank Accounts", min_value=0, step=1)
num_credit_card = st.number_input("Number of Credit Cards", min_value=0, step=1)
interest_rate = st.number_input("Interest Rate", min_value=0.0, step=0.1)
num_of_loan = st.number_input("Number of Loans", min_value=0, step=1)
type_of_loan = st.selectbox("Type of Loan", ["Home Loan", "Car Loan", "Personal Loan", "Credit Card Loan", "Other"])
num_credit_inquiries = st.number_input("Number of Credit Inquiries", min_value=0, step=1)
credit_mix = st.selectbox("Credit Mix", ["Good", "Standard", "Poor"])
outstanding_debt = st.number_input("Outstanding Debt", min_value=0.0, step=100.0)
credit_utilization_ratio = st.number_input("Credit Utilization Ratio", min_value=0.0, step=1.0)
amount_invested_monthly = st.number_input("Amount Invested Monthly", min_value=0.0, step=100.0)
payment_behaviour = st.selectbox("Payment Behaviour", ["Low Spend", "High Spend", "Moderate Spend"])
monthly_balance = st.number_input("Monthly Balance", min_value=0.0, step=100.0)

# Encode categorical variables
occupation_mapping = {"Salaried": 0, "Self-Employed": 1, "Business": 2, "Unemployed": 3}
type_of_loan_mapping = {"Home Loan": 0, "Car Loan": 1, "Personal Loan": 2, "Credit Card Loan": 3, "Other": 4}
credit_mix_mapping = {"Good": 0, "Standard": 1, "Poor": 2}
payment_behaviour_mapping = {"Low Spend": 0, "High Spend": 1, "Moderate Spend": 2}

# Prepare input data for model
input_data = pd.DataFrame({
    'Month': [month],
    'Age': [age],
    'Occupation': [occupation_mapping.get(occupation, -1)],
    'Annual_Income': [annual_income],
    'Monthly_Inhand_Salary': [monthly_inhand_salary],
    'Credit_History_Age': [credit_history_age],
    'Total_Emi_Per_Month': [total_emi_per_month],
    'Num_Bank_Accounts': [num_bank_accounts],
    'Num_Credit_Card': [num_credit_card],
    'Interest_Rate': [interest_rate],
    'Num_Of_Loan': [num_of_loan],
    'Type_Of_Loan': [type_of_loan_mapping.get(type_of_loan, -1)],
    'Num_Credit_Inquiries': [num_credit_inquiries],
    'Credit_Mix': [credit_mix_mapping.get(credit_mix, -1)],
    'Outstanding_Debt': [outstanding_debt],
    'Credit_Utilization_Ratio': [credit_utilization_ratio],
    'Amount_Invested_Monthly': [amount_invested_monthly],
    'Payment_Behaviour': [payment_behaviour_mapping.get(payment_behaviour, -1)],
    'Monthly_Balance': [monthly_balance]
})

# Ensure feature names match the trained model
input_data = input_data.reindex(columns=expected_features, fill_value=0)
st.write("Final input data for model:", input_data)

# Make prediction
if st.button("Predict Credit Score"):
    try:
        prediction = model.predict(input_data)
        st.success(f"Predicted Credit Score: {prediction[0]}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

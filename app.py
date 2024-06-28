import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import csv


fraud_data = {
    "Customer ID": ["CUST001", "CUST002", "CUST003"],
    "Age": [32, 55, 28],
    "Gender": ["Male", "Female", "Male"],
    "Marital Status": ["Married", "Single", "Single"],
    "Dependents": [2, 0, 0],
    "Location": ["ZIP 12345", "City A, State B", "ZIP 54321"],
    "Occupation": ["Professional", "Retired", "Student"],
    "Claims (3yrs)": [1, 0, 2],
    "Claim Type": ["Auto Accident", "None", "Property Damage (Theft)"],
    "Online Activity": ["Frequent", "Low", "High"],
    "Service Calls": [0, 1, 2],
    "Current Insurance": ["Auto, Home", "Auto", "None"],
    "Coverage": ["High", "Medium", "-"],
    "Renewal": ["On-time", "On-time", "Lapsed"],
    "Premium (3yrs)": ["$10,000", "$5,000", "-"],
    "Fraudulent": [1, 0, 1]  
}

# Creating a DataFrame
# df = pd.read_csv("customer_data.csv", dtype=str)
df = pd.DataFrame(fraud_data)

# Encoding categorical variables
label_encoders = {}
for column in ["Gender", "Marital Status", "Location", "Occupation", "Claim Type", "Online Activity", "Current Insurance", "Coverage", "Renewal"]:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    label_encoders[column] = le

# Handle missing values in 'Premium (3yrs)'
df["Premium (3yrs)"] = df["Premium (3yrs)"].str.replace('$', '').str.replace(',', '').replace('-', '0').astype(float)

# Split features and target
X = df.drop(["Customer ID", "Fraudulent"], axis=1)
y = df["Fraudulent"]

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

# Feature scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X, y)

# Function to predict fraudulent claims for new customers
def predict_fraudulent_claim(new_customer_data):
    # Preprocess the new customer data
    for column, le in label_encoders.items():
        new_customer_data[column] = le.transform([new_customer_data[column]])[0]
    
    # Handle the premium value
    new_customer_data["Premium (3yrs)"] = float(new_customer_data["Premium (3yrs)"].replace('$', '').replace(',', ''))

    # Convert to DataFrame
    new_customer_df = pd.DataFrame([new_customer_data])
    
    # Scale the features
    new_customer_df = scaler.transform(new_customer_df)
    
    # Predict using the trained model
    predicted_label = model.predict(new_customer_df)
    return predicted_label[0]

# Streamlit app
st.image("VSoft-Logo.png")
st.title('Insurance Fraud Detection System')

# Input fields for customer data
customer_id = st.text_input('Customer ID')
age = st.number_input('Age', min_value=0)
gender = st.selectbox('Gender', ['Male', 'Female'])
marital_status = st.selectbox('Marital Status', ['Single', 'Married'])
dependents = st.number_input('Dependents', min_value=0)
location = st.text_input('Location')
occupation = st.text_input('Occupation')
claims_3yrs = st.number_input('Claims (3yrs)', min_value=0)
claim_type = st.selectbox('Claim Type', ['None', 'Auto Accident', 'Property Damage (Theft)', 'Other'])
online_activity = st.selectbox('Online Activity', ['Low', 'Medium', 'High' , 'Frequent'])
service_calls = st.number_input('Service Calls', min_value=0)
current_insurance = st.text_input('Current Insurance')
coverage = st.selectbox('Coverage', ['-', 'Low', 'Medium', 'High'])
renewal = st.selectbox('Renewal', ['Lapsed', 'On-time'])
premium_3yrs = st.text_input('Premium (3yrs)')

# Predict button
if st.button('Predict Fraudulent Claim'):
    new_customer_data = {
        "Age": age,
        "Gender": gender,
        "Marital Status": marital_status,
        "Dependents": dependents,
        "Location": location,
        "Occupation": occupation,
        "Claims (3yrs)": claims_3yrs,
        "Claim Type": claim_type,
        "Online Activity": online_activity,
        "Service Calls": service_calls,
        "Current Insurance": current_insurance,
        "Coverage": coverage,
        "Renewal": renewal,
        "Premium (3yrs)": premium_3yrs
    }
    
    is_fraudulent = predict_fraudulent_claim(new_customer_data)
    fraud_status = 'Fraudulent' if is_fraudulent == 1 else 'Not Fraudulent'
    st.write(f'The claim is: {fraud_status}')

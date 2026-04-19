import pickle
import streamlit as st
import pandas as pd

# Load model, feature names, and encoders
with open("customer_churn_model.pkl", "rb") as f:
    model_data = pickle.load(f)

with open("encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

model = model_data["model"]
feature_names = model_data["features_names"]

st.title("Customer Churn Prediction App")

# User inputs - Numeric Features
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
tenure = st.number_input("Tenure (months)", 0, 100, 12)
MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
TotalCharges = st.number_input("Total Charges", 0.0, 10000.0, 500.0)

# User inputs - Categorical Features
gender = st.selectbox("Gender", ["Male", "Female"])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method", [
    "Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"
])

# Preprocessing function
def preprocess_input(user_inputs):
    """Encode categorical features using the trained encoders"""
    processed = {}
    
    # Numeric features - keep as is
    processed["SeniorCitizen"] = user_inputs["SeniorCitizen"]
    processed["tenure"] = user_inputs["tenure"]
    processed["MonthlyCharges"] = user_inputs["MonthlyCharges"]
    processed["TotalCharges"] = user_inputs["TotalCharges"]
    
    # Categorical features - encode using saved encoders
    categorical_cols = ["gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
                       "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
                       "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
                       "PaperlessBilling", "PaymentMethod"]
    
    for col in categorical_cols:
        encoded_value = encoders[col].transform([user_inputs[col]])[0]
        processed[col] = encoded_value
    
    return processed

# Create input DataFrame
if st.button("Predict Churn"):
    user_inputs = {
        "SeniorCitizen": SeniorCitizen,
        "tenure": tenure,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
        "gender": gender,
        "Partner": Partner,
        "Dependents": Dependents,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod
    }
    
    # Preprocess inputs
    preprocessed = preprocess_input(user_inputs)
    
    # Create input DataFrame with correct feature order
    input_data = pd.DataFrame([preprocessed])
    input_data = input_data[feature_names]  # Reorder to match training data
    
    # Prediction
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Customer is likely to churn (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Customer is not likely to churn (Probability: {probability:.2f})")


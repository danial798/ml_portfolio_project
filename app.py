import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and preprocessors
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

st.set_page_config(page_title="CKD Prediction", layout="wide")

st.title("Chronic Kidney Disease (CKD) Prediction")
st.write("Enter the patient's clinical data to predict the risk of CKD.")

# Sidebar for model info
st.sidebar.header("Model Performance")
st.sidebar.write("Model: Random Forest Classifier")
st.sidebar.write("Accuracy: 97.47%")
st.sidebar.write("Precision: 98%")
st.sidebar.write("Recall: 87%")
st.sidebar.write("F1-Score: 92%")
st.sidebar.markdown("---")
st.sidebar.write("Developed by: Atomcamp Student")

# Define features in the correct order (must match training data)
feature_names = [
    'age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 
    'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria', 
    'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium', 
    'potassium', 'hemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 
    'red_blood_cell_count', 'hypertension', 'diabetes_mellitus', 
    'coronary_artery_disease', 'appetite', 'pedal_edema', 'anemia'
]

# Create input fields
inputs = {}
col1, col2, col3 = st.columns(3)

with col1:
    inputs['age'] = st.number_input("Age", min_value=1, max_value=120, value=50)
    inputs['blood_pressure'] = st.number_input("Blood Pressure", min_value=40, max_value=200, value=80)
    inputs['specific_gravity'] = st.selectbox("Specific Gravity", [1.005, 1.010, 1.015, 1.020, 1.025])
    inputs['albumin'] = st.selectbox("Albumin", [0, 1, 2, 3, 4, 5])
    inputs['sugar'] = st.selectbox("Sugar", [0, 1, 2, 3, 4, 5])
    inputs['red_blood_cells'] = st.selectbox("Red Blood Cells", ['normal', 'abnormal'])
    inputs['pus_cell'] = st.selectbox("Pus Cell", ['normal', 'abnormal'])
    inputs['pus_cell_clumps'] = st.selectbox("Pus Cell Clumps", ['notpresent', 'present'])

with col2:
    inputs['bacteria'] = st.selectbox("Bacteria", ['notpresent', 'present'])
    inputs['blood_glucose_random'] = st.number_input("Blood Glucose Random", min_value=0, max_value=500, value=120)
    inputs['blood_urea'] = st.number_input("Blood Urea", min_value=0, max_value=300, value=40)
    inputs['serum_creatinine'] = st.number_input("Serum Creatinine", min_value=0.0, max_value=20.0, value=1.0)
    inputs['sodium'] = st.number_input("Sodium", min_value=0, max_value=200, value=135)
    inputs['potassium'] = st.number_input("Potassium", min_value=0.0, max_value=10.0, value=4.0)
    inputs['hemoglobin'] = st.number_input("Hemoglobin", min_value=0.0, max_value=20.0, value=14.0)
    inputs['packed_cell_volume'] = st.number_input("Packed Cell Volume", min_value=0, max_value=60, value=40)

with col3:
    inputs['white_blood_cell_count'] = st.number_input("White Blood Cell Count", min_value=0, max_value=30000, value=8000)
    inputs['red_blood_cell_count'] = st.number_input("Red Blood Cell Count", min_value=0.0, max_value=10.0, value=5.0)
    inputs['hypertension'] = st.selectbox("Hypertension", ['yes', 'no'])
    inputs['diabetes_mellitus'] = st.selectbox("Diabetes Mellitus", ['yes', 'no'])
    inputs['coronary_artery_disease'] = st.selectbox("Coronary Artery Disease", ['yes', 'no'])
    inputs['appetite'] = st.selectbox("Appetite", ['good', 'poor'])
    inputs['pedal_edema'] = st.selectbox("Pedal Edema", ['no', 'yes'])
    inputs['anemia'] = st.selectbox("Anemia", ['yes', 'no'])

# Prediction logic
if st.button("Predict CKD Risk"):
    try:
        # Prepare input data
        input_data = pd.DataFrame([inputs])
        
        # Encode categorical variables
        for col, le in label_encoders.items():
            if col in input_data.columns:
                # Handle potential unseen labels (though unlikely with selectbox)
                # For simplicity, we assume inputs match training labels
                input_data[col] = le.transform(input_data[col])
        
        # Scale data
        # Ensure columns are in the same order as training
        input_data = input_data[feature_names]
        input_scaled = scaler.transform(input_data)
        
        # Predict
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]
        
        if prediction == 1:
            st.error(f"High Risk of Chronic Kidney Disease (Probability: {probability:.2%})")
        else:
            st.success(f"Low Risk of Chronic Kidney Disease (Probability: {probability:.2%})")
            
    except Exception as e:
        st.error(f"Error making prediction: {e}")

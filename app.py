import streamlit as st
import pandas as pd
import joblib

# Load the dataset and model
df = pd.read_csv('breast_cancer_selected_features.csv')
model = joblib.load('best_ann_model.pkl')

# Streamlit App
st.title('Breast Cancer Prediction App')

# Display Dataset
st.write("### Breast Cancer Dataset")
st.dataframe(df)

# User input
st.write("### User Input Features")
input_data = {}
for col in df.columns[:-1]:
    if col in ['feature_with_categorical_data']:  # Example of handling categorical data
        input_data[col] = st.selectbox(col, df[col].unique())
    else:
        input_data[col] = st.slider(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))

input_df = pd.DataFrame([input_data])

# Prediction
if st.button('Predict'):
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)
    st.write(f"### Prediction: {'Malignant' if prediction[0] == 1 else 'Benign'}")
    st.write(f"### Prediction Probability (Malignant): {prediction_proba[0][1]:.2f}")
    st.write(f"### Prediction Probability (Benign): {prediction_proba[0][0]:.2f}")

# Additional Details
st.write("### Model Information")
st.write("This model is a trained Artificial Neural Network (ANN) with a high accuracy rate in predicting breast cancer. However, it's important to consult with a medical professional for a conclusive diagnosis.")

# Reset Button
if st.button('Reset'):
    st.experimental_rerun()

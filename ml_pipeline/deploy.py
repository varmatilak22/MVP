import streamlit as st
import joblib
import numpy as np

def load_model(model_path):
    return joblib.load(model_path)

model=load_model('../models/linear_regression.pkl')
st.title("Simple Linear Regression Model")

input_value=st.number_input("Enter a Value for Prediction:",min_value=0)

if st.button("Predict"):
    prediction=model.predict(np.array(input_value).reshape(-1,1))
    st.write(f'Prediction:{prediction[0]}')
import streamlit as st
import joblib
import numpy as np
import os

def load_model(model_path):
    return joblib.load(model_path)

deploy_dir=os.path.dirname(__file__)

parent_dir=os.path.dirname(deploy_dir)

models_dir=os.path.join(parent_dir,'models')

model_path=os.path.join(models_dir,'linear_regression.pkl')


print(model_path)
model=load_model(model_path)
st.title("Simple Linear Regression Model")

input_value=st.number_input("Enter a Value for Prediction:",min_value=0)

if st.button("Predict"):
    prediction=model.predict(np.array(input_value).reshape(-1,1))
    st.write(f'Prediction:{prediction[0]}')

import streamlit as st
import joblib
import numpy as np
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pipeline import run_pipeline

def load_model(model_path):
    return joblib.load(model_path)

# Define paths
deploy_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(deploy_dir)
models_dir = os.path.join(parent_dir, 'models')
model_path = os.path.join(models_dir, 'linear_regression.pkl')

# Load the model
model = load_model(model_path)

# Streamlit app
st.set_page_config(page_title="Simple Linear Regression Model", layout="wide")

st.title("Simple Linear Regression Model")

st.markdown("""
    ## Overview
    This application uses a linear regression model to predict the output value based on the input feature value. The model follows the linear relationship:

    \[
    y = mx + c
    \]

    where:
    - \( m \) is the weight (slope) of the line, and
    - \( c \) is the bias (intercept) of the line.
    
    In our model:
    - \( m = 2 \)
    - \( c = 3 \)
    
    Therefore, the equation becomes:
    
    \[
    y = 2x + 3
    \]
""")

# Input feature
input_value = st.number_input("Enter a Value for Prediction:", min_value=0, step=1)

if st.button("Predict"):
    # Predict
    prediction = model.predict(np.array(input_value).reshape(-1, 1))
    st.write(f'**Prediction:** {np.round(prediction[0][0],3)}')

    # Evaluation metrics
    st.markdown("""
        ## Evaluation Metrics
        
        To evaluate the performance of regression models, several metrics are commonly used:

        - **Mean Absolute Error (MAE):** Measures the average magnitude of errors in a set of predictions, without considering their direction.
        - **Mean Squared Error (MSE):** Measures the average of the squares of the errors—i.e., the average squared difference between the estimated values and the actual value.
        - **Root Mean Squared Error (RMSE):** Measures the square root of the average of squared errors, providing an estimate of the standard deviation of the prediction errors.
        - **R-squared (R²):** Represents the proportion of the variance for a dependent variable that's explained by the independent variable(s) in the model.
    """)

    
    result=run_pipeline()

    st.write(f'**Mean Absolute Error (MAE):** {np.round(result,3)}')

# Social media links with icons
st.markdown("""
    ## Connect with Me
    
    Feel free to connect with me on social media:

    - [![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/varmatilak)
    - [![GitHub](https://img.shields.io/badge/GitHub-black?logo=github&logoColor=white)](https://github.com/varmatilak22)
    - [![Kaggle](https://img.shields.io/badge/Kaggle-yellow?logo=kaggle&logoColor=white)](https://www.kaggle.com/xenowing)
""")

import streamlit as st  # Import Streamlit for creating the web app interface
import joblib  # Import joblib for loading the pre-trained model
import numpy as np  # Import numpy for numerical operations
import os  # Import os for operating system functionalities like file path operations
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # Import metrics for evaluating the model
from data_preprocessing import load_data, preprocessing  # Import functions for data loading and preprocessing
from model_evaluation import evaluate_model  # Import function for evaluating the model

# Function to load the pre-trained model
def load_model(model_path):
    """
    Load the model from the specified file path.
    
    Args:
        model_path (str): Path to the saved model file.
    
    Returns:
        model: The loaded model.
    """
    return joblib.load(model_path)

# Define paths to the model file
deploy_dir = os.path.dirname(__file__)  # Get the directory of the current script
parent_dir = os.path.dirname(deploy_dir)  # Get the parent directory of the current script
models_dir = os.path.join(parent_dir, 'models')  # Define the directory where models are stored
model_path = os.path.join(models_dir, 'linear_regression.pkl')  # Define the path to the saved model file

# Load the model
model = load_model(model_path)  # Load the model using the defined path

# Streamlit app configuration
st.set_page_config(page_title="Simple Linear Regression Model", layout="wide")  # Set the configuration for the Streamlit app

st.title("Simple Linear Regression Model")  # Set the title of the app

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
""")  # Provide an overview of the model and its equation in Markdown

# Input feature
input_value = st.number_input("Enter a Value for Prediction:", min_value=0, step=1)  # Create an input box for users to enter a value for prediction

if st.button("Predict"):  # If the 'Predict' button is clicked
    # Predict the output
    prediction = model.predict(np.array(input_value).reshape(-1, 1))  # Predict the output using the model
    st.write(f'**Prediction:** {np.round(prediction[0][0], 3)}')  # Display the rounded prediction

    # Display evaluation metrics
    st.markdown("""
        ## Evaluation Metrics
        
        To evaluate the performance of regression models, several metrics are commonly used:

        - **Mean Absolute Error (MAE):** Measures the average magnitude of errors in a set of predictions, without considering their direction.
        - **Mean Squared Error (MSE):** Measures the average of the squares of the errors—i.e., the average squared difference between the estimated values and the actual value.
        - **Root Mean Squared Error (RMSE):** Measures the square root of the average of squared errors, providing an estimate of the standard deviation of the prediction errors.
        - **R-squared (R²):** Represents the proportion of the variance for a dependent variable that's explained by the independent variable(s) in the model.
    """)  # Explain the evaluation metrics used to assess the model's performance

    # Evaluate the model
    X, y = load_data()  # Load data from the database
    X_train, X_test, y_train, y_test = preprocessing(X, y)  # Preprocess the data for training and testing
    mae, mse, rmse, r2 ,_= evaluate_model(model_path, X_test, y_test)  # Evaluate the model and get metrics

    # Display evaluation metrics
    st.write(f'**Mean Absolute Error (MAE):** {np.round(mae, 3)}')  # Display Mean Absolute Error
    st.write(f'**Mean Squared Error (MSE):** {np.round(mse, 3)}')  # Display Mean Squared Error
    st.write(f'**Root Mean Squared Error (RMSE):** {np.round(rmse, 3)}')  # Display Root Mean Squared Error
    st.write(f'**R-squared (R²):** {np.round(r2, 3)}')  # Display R-squared value

# Social media links with icons
st.markdown("""
    ## Connect with Me
    
    Feel free to connect with me on social media:

    - [![LinkedIn](https://img.shields.io/badge/LinkedIn-blue?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/varmatilak)
    - [![GitHub](https://img.shields.io/badge/GitHub-black?logo=github&logoColor=white)](https://github.com/varmatilak22)
    - [![Kaggle](https://img.shields.io/badge/Kaggle-yellow?logo=kaggle&logoColor=white)](https://www.kaggle.com/xenowing)
""")  # Provide social media links for users to connect

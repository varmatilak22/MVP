from data_preprocessing import load_data, preprocessing  # Import functions for loading and preprocessing data
from model_training import train_model  # Import the function for training the model
from model_evaluation import evaluate_model  # Import the function for evaluating the model

def run_pipeline():
    """
    Execute the end-to-end pipeline for training and evaluating a linear regression model.
    """
    X, y = load_data()  # Load the dataset into features (X) and target values (y)
    X_train, X_test, y_train, y_test = preprocessing(X, y)  # Preprocess the data to create training and testing sets
    train_model(X_train, y_train)  # Train the model using the training data
    result, output = evaluate_model("../models/linear_regression.pkl", X_test, y_test)  # Evaluate the trained model using the test data
    print(f"MAE: {result}, Output: {output}")  # Print the Mean Absolute Error and model predictions

if __name__ == '__main__':
    run_pipeline()  # Execute the pipeline if the script is run directly

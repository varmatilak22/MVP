import joblib  # Import joblib for saving and loading the model
from sklearn.metrics import mean_absolute_error as mae,mean_squared_error as mse,r2_score # Import mean_absolute_error to compute MAE
import matplotlib.pyplot as plt  # Import matplotlib.pyplot for plotting graphs
import seaborn as sns  # Import seaborn for advanced plotting
import numpy as np  # Import numpy for numerical operations
import os  # Import os for operating system functionalities like file path operations

def evaluate_model(model_path, X_test, y_test):
    """
    Load the model from the specified path, make predictions on the test set, 
    and compute the mean absolute error between true and predicted values.
    
    Args:
        model_path (str): Path to the saved model file.
        X_test (numpy.ndarray): Features of the test set.
        y_test (numpy.ndarray): True labels of the test set.
        
    Returns:
        tuple: Mean Absolute Error and predictions.
    """
    try:
        # Load the model from the specified file path
        model = joblib.load(model_path)
    except FileNotFoundError:
        # Handle case where the model file is not found
        print(f"Error: The model file '{model_path}' was not found.")
        return None, None
    
    try:
        # Predict using the loaded model
        predictions = model.predict(X_test)
    except Exception as e:
        # Handle any errors that occur during prediction
        print(f"Error during model prediction: {e}")
        return None, None
    
    # Compute the mean absolute error between true and predicted values
    res1 = mae(y_test, predictions)
    res2 = mse(y_test, predictions)
    res3 = np.sqrt(mse(y_test, predictions))
    res4 = r2_score(y_test, predictions)
    return res1,res2,res3,res4, predictions

def generate_plot(y_test, y_pred, save_path='sample.png'):
    """
    Generate a scatter plot to compare actual vs predicted values and save the plot.
    
    Args:
        y_test (numpy.ndarray): True labels of the test set.
        y_pred (numpy.ndarray): Predicted values from the model.
        save_path (str): Path to save the generated plot.
        
    Returns:
        None
    """
    # Create an array of indices for x-axis
    X = np.arange(0, len(y_test), dtype=int)
    # Flatten and sort true and predicted values for better visualization
    y_test_flat = np.sort(y_test.flatten())
    y_pred_flat = np.sort(y_pred.flatten())
    
    # Create a figure with a specific size
    plt.figure(figsize=(8, 6))
    # Plot true values as blue scatter points
    sns.scatterplot(x=X, y=y_test_flat, color='blue', label='True Label', s=50)
    # Plot predicted values as green scatter points
    sns.scatterplot(x=X, y=y_pred_flat, color='green', label='Predicted', s=50)
    
    # Set the labels for the x and y axes
    plt.xlabel("Sample Index")
    plt.ylabel("Value")
    # Set the title of the plot
    plt.title("Actual vs Predicted Values")
    # Display the legend
    plt.legend()
    # Add a grid to the plot for better readability
    plt.grid(True)
    
    # Save the plot to the specified file path
    plt.savefig(save_path)
    # Display the plot
    plt.show()

if __name__ == '__main__':
    # Import functions for data preprocessing
    from data_preprocessing import load_data, preprocessing
    
    # Load and preprocess the data
    X, y = load_data()
    X_train, X_test, y_train, y_test = preprocessing(X, y)
    
    # Define the path to the saved model file
    model_path = '../models/linear_regression.pkl'
    # Evaluate the model using the test data
    evaluate_model(model_path, X_test, y_test)
    
    
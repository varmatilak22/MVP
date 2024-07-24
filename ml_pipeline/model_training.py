from sklearn.linear_model import LinearRegression  # Import the LinearRegression class from sklearn for creating and training the linear regression model
import joblib  # Import joblib for saving the trained model to a file

def train_model(X_train, y_train):
    """
    Train a linear regression model using the provided training data and save it to a file.

    Args:
        X_train (array-like): Training features.
        y_train (array-like): Training target values.
    """
    model = LinearRegression()  # Create an instance of the LinearRegression model
    model.fit(X_train, y_train)  # Fit the model to the training data
    joblib.dump(model, '../models/linear_regression.pkl')  # Save the trained model to a file using joblib

if __name__ == '__main__':
    from data_preprocessing import load_data, preprocessing  # Import functions for loading and preprocessing data
    X, y = load_data()  # Load the dataset
    X_train, X_test, y_train, y_test = preprocessing(X, y)  # Preprocess the data to create training and testing sets
    train_model(X_train, y_train)  # Train the model using the training data and save it

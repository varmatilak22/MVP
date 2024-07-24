import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split
import os

def load_data():
    # Construct the path to the database file dynamically
    root_dir = os.path.dirname(__file__)
    sub_dir = os.path.dirname(root_dir)
    sub_dir_2 = os.path.join(sub_dir, 'data')
    db_path = os.path.join(sub_dir_2, 'linear.db')
    print(db_path)

    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Initialize lists to hold the feature and target values
    X = []
    y = []

    # Execute SQL query to fetch all data from the 'data' table
    cursor.execute("SELECT * FROM data;")
    for i, j in cursor.fetchall():
        X.append(i)
        y.append(j)
    
    conn.commit()
    conn.close()
    return X, y

# Load the data
X, y = load_data()
print(f"X: {X}")
print(f"y: {y}")

def preprocessing(x, y):
    # Convert lists to numpy arrays and reshape
    x = np.array(x).reshape(-1, 1)
    y = np.array(y).reshape(-1, 1)
    # Split data into training and testing sets
    return train_test_split(x, y, test_size=0.2, random_state=1)

# Preprocess the data
X_train, X_test, y_train, y_test = preprocessing(X, y)
print(f"X_train: {X_train}")
print(f"X_test: {X_test}")
print(f"y_train: {y_train}")
print(f"y_test: {y_test}")

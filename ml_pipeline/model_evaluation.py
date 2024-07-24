import joblib 
from sklearn.metrics import mean_absolute_error as mae
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model(model_path,X_test,y_test):
    model=joblib.load(model_path)
    predict=model.predict(X_test)
    result=mae(y_test,predict)
    return result,predict

def generate_plot(y_test,y_pred):
    """
    function will generate a plot for linear regression it will be scatter plot
    """
    X=np.arange(0,len(y_test),dtype=int)
    y_test_flat=np.sort(y_test.flatten())
    y_pred_flat=np.sort(y_pred.flatten())
    plt.figure(figsize=(6,6))
    sns.scatterplot(x=X,y=y_test_flat,color='blue',label='True label')
    sns.scatterplot(x=X,y=y_pred_flat,color='green',label='Predicted')
    plt.xlabel("Input value")
    plt.ylabel("Prediction")
    plt.title("Actual Vs Predicted")
    plt.legend()
    plt.savefig('sample.png')
    plt.show()

#generate_plot([1,2,3,4,5],[2,4,5,6,7])



if __name__=='__main__':
    from data_preprocessing import load_data,preprocessing
    X,y=load_data()
    X_train,X_test,y_train,y_test=preprocessing(X,y)
    result,predict=evaluate_model('../models/linear_regression.pkl',X_test,y_test)
    generate_plot(y_test,predict)
    print(f"Mean Squared Error:{result}")

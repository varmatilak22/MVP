from data_preprocessing import load_data,preprocessing
from model_training import train_model
from model_evaluation import evaluate_model

def run_pipeline():
    X,y=load_data()
    X_train,X_test,y_train,y_test=preprocessing(X,y)
    train_model(X_train,y_train)
    result,output=evaluate_model("../models/linear_regression.pkl",X_test,y_test)
    print(f"Cost Function:{result}")
    print(f"Predicted Output:{output}")

if __name__=='__main__':
    run_pipeline()
import joblib 
from sklearn.metrics import mean_absolute_error as mae


def evaluate_model(model_path,X_test,y_test):
    model=joblib.load(model_path)
    predict=model.predict(X_test)
    result=mae(y_test,predict)
    return result,predict


if __name__=='__main__':
    from data_preprocessing import load_data,preprocessing
    X,y=load_data()
    X_train,X_test,y_train,y_test=preprocessing(X,y)
    result,predict=evaluate_model('../models/linear_regression.pkl',X_test,y_test)
    print(f"Mean Squared Error:{result},Output:{predict}")
from sklearn.linear_model import LinearRegression
import joblib

def train_model(X_train,y_train):
    model=LinearRegression()
    model.fit(X_train,y_train)
    joblib.dump(model,'../models/linear_regression.pkl')

if __name__=='__main__':
    from data_preprocessing import load_data,preprocessing
    X,y=load_data()
    X_train,X_test,y_train,y_test=preprocessing(X,y)
    train_model(X_train,y_train)




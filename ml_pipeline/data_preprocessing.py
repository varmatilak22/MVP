import numpy as np
import sqlite3
from sklearn.model_selection import train_test_split

def load_data():
    conn=sqlite3.connect("C:\Py\programs\linear_regression\data\linear.db")
    cursor=conn.cursor()

    X=[]
    y=[]

    cursor.execute("select * from data;")
    for i,j in cursor.fetchall():
       X.append(i),y.append(j)
    
    conn.commit()
    conn.close()
    return X,y

X,y=load_data()
print(f"X:{X}")
print(f"y:{y}")

def preprocessing(x,y):
    x=np.array(x).reshape(-1,1)
    y=np.array(y).reshape(-1,1)
    return train_test_split(x,y,test_size=0.2,random_state=1)


X_train,X_test,y_train,y_test=preprocessing(X,y)
print(f"X_train:{X_train}")
print(f"X_test:{X_test}")
print(f"y_train:{y_train}")
print(f"y_test:{y_test}")



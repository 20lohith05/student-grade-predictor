import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

def train_model():
    df=pd.read_csv("data/data.csv")
    df["sex"]=df["sex"].map({"M":1,"F":0})

    x=df[["age","sex",
          "studytime","failures",
          "absences","G1","G2"]]
    
    y=df["G3"]

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=99)
    
    model=LinearRegression()
    model.fit(x_train,y_train)

    y_pred=model.predict(x_test)
    mae=mean_absolute_error(y_test,y_pred)
    rmse=np.sqrt(mean_squared_error(y_test,y_pred))
    r2=r2_score(y_test,y_pred)

    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.2f}")

    return model
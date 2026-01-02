import pandas as pd
from sklearn.linear_model import LinearRegression

def train_model():
    df=pd.read_csv("data/data.csv")
    df["sex"]=df["sex"].map({"M":1,"F":0})

    x=df[["age","sex",
          "studytime","failures",
          "absences","G1","G2"]]
    
    y=df["G3"]

    model=LinearRegression()
    model.fit(x,y)

    return model

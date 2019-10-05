from joblib import dump,load
import pandas as pd
import numpy as np
model=load('Dracula.mdl')

df=pd.read_csv('data.csv')
x_test=df.drop("MEDV",axis=1)
x_test=x_test.iloc[:5]
x_label=df["MEDV"].iloc[:5].copy()

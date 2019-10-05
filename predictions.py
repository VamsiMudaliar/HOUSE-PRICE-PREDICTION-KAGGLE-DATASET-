# HOUSE PRICE PREDICTION USING VARIOUS MODELS 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from joblib import dump,load
#MODELS
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#READING CSV FILE
df=pd.read_csv('data.csv')

#print(df.describe()) describes min,count,avg,max,std etc...


#this Function Takes Care of missing values if any in our dataset we will use simpleImputer to achieve this thing
def handle_missing_values(df):
    imputer=SimpleImputer(strategy="median")
    X=imputer.fit_transform(df)
    return pd.DataFrame(X,columns=df.columns)


train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
print(test_set.shape)
print(train_set.shape)
df=handle_missing_values(df)

#To Distribute Data Evenly in train as well as in test data
sp=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index,test_index in sp.split(df,df["CHAS"]):
    strat_train_set=df.loc[train_index]
    strat_test_set=df.loc[test_index]
#COPYING NEW SET IN OUR ORIGNAL DATAFRAME NOW THE DATAFRAME CONTAINS TRAINING DATA
df=strat_train_set.copy()

#FUNCTION TO CALCULATE CORELATION
def calculate_corelation(df):
    cor_matrix=df.corr()
    print(cor_matrix["MEDV"].sort_values(ascending=False))


#NOW WE CREATE OUR PIPLINE WHERE WHENEVER A NEW DATA COMES IT GOES THORUGH THAT PIPELINE A GETS NORMALIZED 
mypipeline=Pipeline([("imputer",SimpleImputer(strategy="median")),("standardScalar",StandardScaler())])

#DROP THE LAST MEDV AND GIVE IT TO AS LABEL
df=strat_train_set.drop("MEDV",axis=1)
df_labels=strat_train_set["MEDV"].copy()

#PASS OUR NEW CONSTRUCTED DATA FRAME INTO THE PIPLINE
new_df=mypipeline.fit_transform(df)

#CHOOSE A MODEL
model=RandomForestRegressor()
model.fit(new_df,df_labels)

#MAKING SOME DATA 5 from features and corresponding 5 from labels
dummy_data=df.iloc[:5]
dummy_data_labels=df_labels.iloc[:5]
w=mypipeline.fit_transform(dummy_data)
#PREDICTION
result=model.predict(w)
#OUTPUT ACTUAL AND PREDICTED VALUE
actual_values=list(dummy_data_labels)
print(result,actual_values)

#EVALUATION OF MY MODEL

mse=mean_squared_error(actual_values,result)
rmse=np.sqrt(mse)
print("ROOT MEAN SQUARED ERROR :",rmse)

#USING CROSS VALIDATION TECHNIQUE

score=cross_val_score(model,new_df,df_labels,scoring="neg_mean_squared_error",cv=10)
validate=np.sqrt(-score)

#FUNCTION TO PRINT MEAN AND STD 
def model_score(z):
    print("Mean  :",z.mean())
    print("Standard Deviation :",z.std())
model_score(validate)

#TEST
'''
li=[[1.05393,	0,	8.14,	0,	0.538,	5.935,	29.3,	4.4986,	4	,307	,21,	386.85	,6.58]]
new_li=mypipeline.fit_transform(li)
print("Answer ",model.predict(new_li))
'''
#SERiALIZE THE MODEL

dump(model,'Dracula.mdl')
print("Model Saved")






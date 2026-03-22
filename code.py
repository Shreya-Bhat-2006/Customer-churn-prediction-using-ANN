import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler


 
df=pd.read_csv("Data.csv")
df.drop('customerID',axis="columns",inplace=True)
df=df[df.TotalCharges!=" "]
df.TotalCharges=pd.to_numeric(df.TotalCharges)
df.replace('No internet service','No',inplace=True)
df.replace('No phone service','No',inplace=True)

def print_unique_col_values(df):
    for column in df:
        if df[column].dtypes == 'object':
            print(f'{column} : {df[column].unique()}')

yes_no_columns = ['Partner','Dependents','PhoneService','MultipleLines', 'OnlineSecurity','OnlineBackup',
'DeviceProtection','TechSupport','StreamingTV','StreamingMovies' ,'PaperlessBilling' ,'Churn']


for col in yes_no_columns:
    df[col] = df[col].map({"Yes": 1, "No": 0})

df["gender"] = df["gender"].map({"Male": 0, "Female": 1})
df = pd.get_dummies(df, columns=["InternetService", "Contract", "PaymentMethod"], dtype=int)

col_to_scale=["tenure","MonthlyCharges","TotalCharges"]

Scalar=MinMaxScaler()
df[col_to_scale]=Scalar.fit_transform(df[col_to_scale])
print(df.head())
import pandas as pd
from matplotlib import cm, pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sn

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




X=df.drop("Churn",axis="columns")
Y=df["Churn"]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=5)

model=keras.Sequential(
    [
        keras.layers.Dense(20,input_shape=(X_train.shape[1],),activation="relu"),
        keras.layers.Dense(1,activation="sigmoid")
    ]
)

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
model.fit(X_train,Y_train,epochs=100)

model.evaluate(X_test,Y_test)

Y_pred = (model.predict(X_test) > 0.5).astype(int)
Y_pred = Y_pred.flatten()

print(classification_report(Y_test, Y_pred))
cm = confusion_matrix(Y_test, Y_pred)

plt.figure(figsize=(10,7))
sn.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
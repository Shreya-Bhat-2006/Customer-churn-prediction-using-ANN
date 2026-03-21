import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

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

# Call the function with your dataframe
print_unique_col_values(df)

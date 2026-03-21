import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

df=pd.read_csv("Data.csv")
df.drop('customerID',axis="columns",inplace=True)
df=df[df.TotalCharges!=" "]
df.TotalCharges=pd.to_numeric(df.TotalCharges)

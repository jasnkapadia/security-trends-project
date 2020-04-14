import numpy as np 
import pandas as pd
from os import listdir 

df = pd.read_csv("cyber-security-breaches.csv")
df = df.drop('Unnamed: 0', axis=1)

print(df)
import pandas as pd
from data_handling import computeRUL
df_train1 = pd.read_csv("train_FD001.csv")
df_train1=computeRUL(df_train1)
df_train1.to_csv("train_FD001_with_rul.csv", index=False)
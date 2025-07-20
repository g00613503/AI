import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def get_iris_data_path():
    current_path = os.getcwd()
    print(f"当前所在目录为：{current_path}")
    data_path = os.path.join(os.path.dirname(current_path),"database","Iris","Iris.csv")
    print(f"文件所在路径为：{data_path}")
    return  data_path

def load_path(data_path):
    df = pd.read_csv(data_path,delimiter=",")
    nRow,nCol = df.shape
    print(f"数据集的行数为：{nRow}，列数为{nCol}")
    print(df.head())

    X = df.iloc[:, 1:5].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    Y = df.iloc[:,5:].values

    return X_scaled,Y

def Get_Iris_data():
    data_path = get_iris_data_path()
    return load_path(data_path)
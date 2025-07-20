from database.Iris.Iris import Get_Iris_data
from sklearn. cluster import KMeans
import numpy as np
from sklearn.metrics import mean_squared_error

def k_means(X_scaled,y_label):
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    y_pred = kmeans.labels_

    a = y_pred.ravel()
    squared_errors = mean_squared_error(y_pred.ravel(),y_label.ravel())
    # 计算总平方差 (SSE)
    sse = np.sum(squared_errors)
    # 计算均方误差 (MSE)
    mse = np.mean(squared_errors)
    print(f"总方差为{sse},均方差为{mse}")



if __name__ == '__main__':
    # 加载鸢尾花数据集（分类任务）
    X_scaled,y_label = Get_Iris_data()
    k_means(X_scaled,y_label)
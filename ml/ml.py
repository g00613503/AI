from database.Iris.Iris import Get_Iris_data
from sklearn. cluster import KMeans
from sklearn.metrics import silhouette_score

def k_means(data):
    inertia = []
    silhouette_scores = []
    K_range = range(2, 11)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data)
        inertia.append(kmeans.inertia_)
        if k > 1:
            silhouette_scores.append(silhouette_score(data, kmeans.labels_))

if __name__ == '__main__':
    # 加载鸢尾花数据集（分类任务）
    data,target = Get_Iris_data()
    k_means(data)
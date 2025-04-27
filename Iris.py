import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# Train KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Define a function to predict cluster
def predict_cluster(input_features):
    """
    input_features: list of 4 numbers [sepal length, sepal width, petal length, petal width]
    returns: cluster number
    """
    df = pd.DataFrame([input_features], columns=iris.feature_names)
    cluster = kmeans.predict(df)[0]
    return int(cluster)

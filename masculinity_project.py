
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset
survey = pd.read_csv("masculinity.csv")

# Display initial data exploration
print(survey.columns)
print(len(survey))
print(survey["q0007_0001"].value_counts())
print(survey.head())

# Map survey responses to numerical values
cols_to_map = ["q0007_0001", "q0007_0002", "q0007_0003", "q0007_0004",
               "q0007_0005", "q0007_0006", "q0007_0007", "q0007_0008", 
               "q0007_0009", "q0007_0010", "q0007_0011"]
for cols in cols_to_map:
    survey[cols] = survey[cols].map({"Often": 4, "Sometimes": 3, "Rarely": 2, 
                                     "Never, but open to it": 1, 
                                     "Never, and not open to it": 0})

# Verify the mapping
print(survey["q0007_0001"].value_counts())

# Scatter plot visualization
plt.scatter(survey["q0007_0001"], survey["q0007_0002"], alpha=0.1)
plt.xlabel("Ask a friend for professional advice")
plt.ylabel("Ask a friend for personal advice")
plt.show()

# Prepare data for clustering
rows_to_cluster = survey.dropna(subset=["q0007_0001", "q0007_0002", "q0007_0003", 
                                        "q0007_0004", "q0007_0005", "q0007_0008", 
                                        "q0007_0009"])

# Perform KMeans clustering
classifier = KMeans(n_clusters=2)
classifier.fit(rows_to_cluster[["q0007_0001", "q0007_0002", "q0007_0003", 
                                "q0007_0004", "q0007_0005", "q0007_0008", 
                                "q0007_0009"]])

# Display cluster centers
print(classifier.cluster_centers_)

# Analyze cluster membership
print(classifier.labels_)
cluster_zero_indices = []
cluster_one_indices = []
for i in range(len(classifier.labels_)):
    if classifier.labels_[i] == 0:
        cluster_zero_indices.append(i)
    elif classifier.labels_[i] == 1:
        cluster_one_indices.append(i)

# Create DataFrames for each cluster
cluster_zero_df = rows_to_cluster.iloc[cluster_zero_indices]
cluster_one_df = rows_to_cluster.iloc[cluster_one_indices]

# Analyze educational levels within clusters
print(cluster_zero_df['educ4'].value_counts() / len(cluster_zero_df))
print(cluster_one_df['educ4'].value_counts() / len(cluster_one_df))

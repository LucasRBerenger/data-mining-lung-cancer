import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn import tree
from sklearn import svm
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score, rand_score, adjusted_rand_score, normalized_mutual_info_score, adjusted_mutual_info_score

table = pd.read_csv(">> SET YOUR FILE PATH HERE <<", encoding="latin1")

data = table.drop(['id', 'diagnosis'], axis=1)
labels = table['diagnosis'].map({'M': 1, 'B': 0})

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.25, random_state=0)

def accuracy(confusion_matrix):
    sum, total = 0, 0
    for i in range(len(confusion_matrix)):
        for j in range(len(confusion_matrix[0])):
            if i == j:
                sum += confusion_matrix[i,j]
            total += confusion_matrix[i,j]
    return sum/total

sns.countplot(x='diagnosis', data=table)
#plt.title('Diagnosis Distribution')
#plt.xlabel('Diagnosis (0: Benign, 1: Malignant)')
#plt.ylabel('Count')
#plt.show()

print('')
print('')
print('======================Linear Regression==================================')
regression = LinearRegression()
regression.fit(train_data, train_labels)
predictions = regression.predict(test_data)

mae = mean_absolute_error(test_labels, predictions)
mse = mean_squared_error(test_labels, predictions)
r2 = r2_score(test_labels, predictions)

# Mean Absolute Error Calculation
print('MAE: %.2f' % mae)
# Mean Squared Error Calculation
print('Mean squared error: %.2f' % mse)
# R2 Score Calculation
print('R2 Score: %.2f' % r2)

print('')
print('')
print('======================Decision Tree Regression==========================')
tree_model = tree.DecisionTreeRegressor()
tree_model.fit(train_data, train_labels)
predictions = tree_model.predict(test_data)

print('MAE: %.2f' % mean_absolute_error(test_labels, predictions))
print('Mean squared error: %.2f' % mean_squared_error(test_labels, predictions))
print('R2 Score: %.2f' % r2_score(test_labels, predictions))

print('')
print('')
print('======================SVM Regression====================================')
svm_model = svm.SVR()
svm_model.fit(train_data, train_labels)
predictions = svm_model.predict(test_data)

print('MAE: %.2f' % mean_absolute_error(test_labels, predictions))
print('Mean squared error: %.2f' % mean_squared_error(test_labels, predictions))
print('R2 Score: %.2f' % r2_score(test_labels, predictions))

print('')
print('')
print('======================DBSCAN Clustering===============================')

selected_columns = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean'
]

clustering_data = table[selected_columns]

# Preprocessing - Standardizing the variables to the same scale
ss = StandardScaler()
scaled_data = ss.fit_transform(clustering_data)
scaled_clustering_data = pd.DataFrame(scaled_data, columns=selected_columns)

print('Data after preprocessing:\n', scaled_clustering_data.head())

#eps = maximum distance for two points to be neighbors
#min_samples = minimum number of points to form a cluster
dbs = DBSCAN(eps=0.8, min_samples=4)
dbs.fit(scaled_clustering_data)

#Labels, if -1 is returned it is noise
labels = dbs.labels_
scaled_clustering_data['labels'] = labels
noise = list(labels).count(-1)
print('')
print("Number of noise points:", noise)

# Number of clusters
n_labels = len(np.unique(labels)) - (1 if -1 in labels else 0)
print("Number of clusters:", n_labels)
print('')
plt.figure(figsize=(10, 6))
sns.scatterplot(data=scaled_clustering_data,
                x='radius_mean', y='texture_mean',
                hue='labels', palette='muted', legend='full').set_title('Clusters found by DBSCAN')
plt.xlabel('Mean Radius')
plt.ylabel('Mean Texture')
plt.show()

# Calculate silhouette coefficient
print('')
if n_labels > 1:  # If there's only one cluster, there's no silhouette coefficient
    s_score = silhouette_score(scaled_clustering_data.iloc[:, :-1], labels)
    print(f"Silhouette Coefficient: {s_score:.3f}")
else:
    print("Error...")

print('')
print('')
print('======================K-Means Clustering==============================')
print('')
print('')

ss = StandardScaler()
scaled_data = ss.fit_transform(clustering_data)
scaled_kmeans_data = pd.DataFrame(scaled_data, columns=selected_columns)

v_measure_score_evaluation = []
rand_score_evaluation = []
adjusted_rand_score_evaluation = []
normalized_mutual_info_score_evaluation = []
adjusted_mutual_info_score_evaluation = []

for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, n_init="auto").fit(scaled_kmeans_data)
    v_measure_score_evaluation.append(v_measure_score(kmeans.labels_, kmeans.fit_predict(scaled_kmeans_data)))
    rand_score_evaluation.append(rand_score(kmeans.labels_, kmeans.fit_predict(scaled_kmeans_data)))
    adjusted_rand_score_evaluation.append(adjusted_rand_score(kmeans.labels_, kmeans.fit_predict(scaled_kmeans_data)))
    normalized_mutual_info_score_evaluation.append(normalized_mutual_info_score(kmeans.labels_, kmeans.fit_predict(scaled_kmeans_data)))
    adjusted_mutual_info_score_evaluation.append(adjusted_mutual_info_score(kmeans.labels_, kmeans.fit_predict(scaled_kmeans_data)))

plt.figure(figsize=(8, 6), dpi=100)
plt.plot(range(2, 8), v_measure_score_evaluation, color="green", marker="o")
plt.plot(range(2, 8), rand_score_evaluation, color="blue", marker="o")
plt.plot(range(2, 8), adjusted_rand_score_evaluation, color="red", marker="o")
plt.plot(range(2, 8), normalized_mutual_info_score_evaluation, color="pink", marker="o")
plt.plot(range(2, 8), adjusted_mutual_info_score_evaluation, color="black", marker="o")
plt.xlabel("Number of clusters (K)")
plt.ylabel("Evaluation Metrics")
plt.legend(['v_measure_score', 'rand_score', 'adjusted_rand_score', 'normalized_mutual_info_score', 'adjusted_mutual_info_score'])
plt.title('K-Means Evaluation with Different K')
plt.show()

print('')
print('')

kmeans = KMeans(n_clusters=3, n_init="auto").fit(scaled_kmeans_data)
scaled_kmeans_data['labels'] = kmeans.labels_

plt.figure(figsize=(10, 6))
sns.scatterplot(data=scaled_kmeans_data, x='radius_mean', y='texture_mean', hue='labels', palette='muted', legend='full')
plt.title('Clusters Generated with K-Means (K=3)')
plt.xlabel('Mean Radius (Standardized)')
plt.ylabel('Mean Texture (Standardized)')
plt.show()

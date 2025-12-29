# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 13:47:18 2025
@author: Neal
"""
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np


# Ignore all warnings
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['OMP_NUM_THREADS'] = '1'

iris = load_iris()
X, y = iris.data[:,[1,3]], iris.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k_values = (2, 3, 4, 5, 6, 7, 8, 9, 10)
eps_values = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
min_samples_values = (3, 5, 7, 9)

# %%
#++insert your code below++

# ==================== PART 1: KMeans Clustering ====================
print("="*70)
print("KMEANS CLUSTERING ANALYSIS")
print("="*70)

kmeans_results = []

for k in k_values:
    # Create KMeans model with specified parameters
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    
    # Fit the model
    labels = kmeans.fit_predict(X_scaled)
    
    # Calculate inertia
    inertia = kmeans.inertia_
    
    # Calculate silhouette score
    sil_score = silhouette_score(X_scaled, labels)
    
    # Store results
    kmeans_results.append({
        'k': k,
        'inertia': inertia,
        'silhouette_score': sil_score
    })
    
    print(f"K={k}: Inertia={inertia:.4f}, Silhouette Score={sil_score:.4f}")

print()

# Find best k based on silhouette score
best_kmeans = max(kmeans_results, key=lambda x: x['silhouette_score'])
print(f"Best K-Means configuration: K={best_kmeans['k']} with Silhouette Score={best_kmeans['silhouette_score']:.4f}")
print()

# ==================== PART 2: DBSCAN Clustering ====================
print("="*70)
print("DBSCAN CLUSTERING ANALYSIS")
print("="*70)

dbscan_results = []

for eps in eps_values:
    for min_samples in min_samples_values:
        # Create DBSCAN model
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        
        # Fit the model
        labels = dbscan.fit_predict(X_scaled)
        
        # Count number of clusters (excluding noise which is labeled as -1)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        # Count noise points
        n_noise = list(labels).count(-1)
        
        # Calculate silhouette score only if there are at least 2 clusters
        # and not all points are noise
        if n_clusters >= 2 and n_noise < len(labels):
            try:
                sil_score = silhouette_score(X_scaled, labels)
            except:
                sil_score = -1  # Invalid score
        else:
            sil_score = -1  # Invalid configuration
        
        # Store results
        dbscan_results.append({
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette_score': sil_score
        })
        
        print(f"eps={eps}, min_samples={min_samples}: Clusters={n_clusters}, Noise={n_noise}, Silhouette={sil_score:.4f}")

print()

# Find best DBSCAN configuration based on silhouette score
valid_dbscan_results = [r for r in dbscan_results if r['silhouette_score'] > -1]

if valid_dbscan_results:
    best_dbscan = max(valid_dbscan_results, key=lambda x: x['silhouette_score'])
    print(f"Best DBSCAN configuration:")
    print(f"  eps={best_dbscan['eps']}, min_samples={best_dbscan['min_samples']}")
    print(f"  Number of clusters: {best_dbscan['n_clusters']}")
    print(f"  Number of noise points: {best_dbscan['n_noise']}")
    print(f"  Silhouette Score: {best_dbscan['silhouette_score']:.4f}")
else:
    print("No valid DBSCAN configuration found!")

print()

# ==================== ANSWERS TO QUESTIONS ====================
print("="*70)
print("ANSWERS TO QUESTIONS")
print("="*70)

# Question 1: How does inertia change when number of clusters increases?
print("\nQuestion 1: When the number of clusters increases, how will the inertia change?")
print("Answer: The inertia DECREASES monotonically as the number of clusters increases.")
print("Explanation: More clusters mean points are closer to their centroids, reducing inertia.")
print("\nInertia values:")
for result in kmeans_results:
    print(f"  K={result['k']}: Inertia={result['inertia']:.4f}")

# Question 2: Which k gives the highest silhouette score?
print(f"\nQuestion 2: What is the number of clusters that yields the highest silhouette score?")
print(f"Answer: K = {best_kmeans['k']}")
print(f"  Silhouette Score = {best_kmeans['silhouette_score']:.4f}")

# Question 3: Number of valid clusters for best DBSCAN
if valid_dbscan_results:
    print(f"\nQuestion 3: What is the number of valid clusters for the clustering results with the highest silhouette score (DBSCAN)?")
    print(f"Answer: {best_dbscan['n_clusters']} clusters")
    print(f"  (eps={best_dbscan['eps']}, min_samples={best_dbscan['min_samples']})")

    # Question 4: Number of noise examples for best DBSCAN
    print(f"\nQuestion 4: What is the number of noise examples for the clustering results with the highest silhouette score (DBSCAN)?")
    print(f"Answer: {best_dbscan['n_noise']} noise points")
    print(f"  (eps={best_dbscan['eps']}, min_samples={best_dbscan['min_samples']})")

print("\n" + "="*70)
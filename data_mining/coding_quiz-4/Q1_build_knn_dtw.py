# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 19:05:16 2025
@author: Neal LONG
Task: Time Series Classification with Modified Dynamic Time Warping (DTW)
"""
import numpy as np
import pickle

def dtw_dp(s, t):
    """
    Compute DTW distance between s and t, with the cost of 
    matching two values a and b being the absolute difference between them, 
    i.e.,  cost(a,b) = abs(a-b).
    
    Parameters:
    s, t: input time series (1D arrays)
    
    Returns:
    DTW distance (float)
    """
    # PART A: DTW Dynamic Programming Implementation
    n, m = len(s), len(t)
    
    # Initialize DP matrix with infinity
    dp = np.full((n + 1, m + 1), np.inf)
    dp[0, 0] = 0  # Base case: distance between two empty sequences is 0
    
    # Fill the DP matrix
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            # Cost of matching s[i-1] and t[j-1] using absolute difference
            cost = abs(s[i-1] - t[j-1])
            
            # Recurrence relation: take minimum of three directions
            dp[i, j] = cost + min(
                dp[i-1, j],      # insertion (move up)
                dp[i, j-1],      # deletion (move left)
                dp[i-1, j-1]     # match (move diagonal)
            )
    
    # Return DTW distance
    return dp[n, m]


class KNN_DTW():
    def __init__(self, k=5):
        self.k = k
        self.dist_func = dtw_dp
        
    def fit(self, X, y):
        """Simply store training data"""
        if len(X) != len(y):
            raise ValueError("X and y must have same length")
        self.X_train = X
        self.y_train = y
        return self
    
    def predict_proba(self, x_test):
        """
        Predict label and probability for one test sample
        
        Parameters:
        x_test: A single time series record (1D array)
        
        Returns:
        (predicted_label, probability)
        """
        # PART B: KNN with DTW and Inverse Logarithm Weighting
        
        # Step 1: Compute distances to all training samples using DTW
        distances = []
        for x_train in self.X_train:
            dist = self.dist_func(x_test, x_train)
            distances.append(dist)
        
        distances = np.array(distances)
        
        # Step 2: Get indices of k nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Step 3: Get distances and labels of k nearest neighbors
        k_distances = distances[k_indices]
        k_labels = [self.y_train[i] for i in k_indices]
        
        # Step 4: Apply inverse logarithm weighting
        # weight = 1 / log(distance + epsilon)
        weights = 1.0 / np.log(k_distances + 1e-8)
        
        # Step 5: Calculate weighted votes for each class
        unique_labels = list(set(k_labels))
        label_weights = {label: 0.0 for label in unique_labels}
        
        for i, label in enumerate(k_labels):
            label_weights[label] += weights[i]
        
        # Step 6: Find the label with maximum weight
        total_weight = sum(label_weights.values())
        predicted_label = max(label_weights, key=label_weights.get)
        probability = label_weights[predicted_label] / total_weight
        
        return predicted_label, probability


if __name__ == "__main__":
    # Load data
    with open("./data/ts_data.pkl", 'rb') as rf:
        ts_data, y, test_ts = pickle.load(rf)
    
    print(f"Dataset: {len(ts_data)} training samples")
    print(f"Type of ts_data[0]: {type(ts_data[0])}, Length: {len(ts_data[0])}")
    print(f"Type of test_ts: {type(test_ts)}, Length: {len(test_ts)}")
    print(f"# class labels: {len(set(y))}")
    print(f"Class labels: {sorted(set(y))}")
    print()
    
    # ========== PART C: Test DTW and Make Predictions ==========
    
    # Test DTW between first two training samples
    dist_01 = dtw_dp(ts_data[0], ts_data[1])
    print(f"DTW distance between ts_data[0] and ts_data[1]: {dist_01:.6f}")
    print()
    
    # Question 1: DTW distance between 3rd time series and test_ts
    # Note: 3rd time series is at index 2 (0-indexed)
    dist_3_test = dtw_dp(ts_data[2], test_ts)
    print("=" * 60)
    print("ANSWER TO QUESTION 1:")
    print(f"DTW distance between ts_data[2] (3rd time series) and test_ts: {dist_3_test:.6f}")
    print("=" * 60)
    print()
    
    # Train KNN_DTW model
    print("Training KNN_DTW model with k=5...")
    knn_model = KNN_DTW(k=5)
    knn_model.fit(ts_data, y)
    print("Model trained successfully!")
    print()
    
    # Question 2 & 3: Predict label and probability for test_ts (ts_new)
    predicted_label, probability = knn_model.predict_proba(test_ts)
    
    print("=" * 60)
    print("ANSWER TO QUESTION 2:")
    print(f"Predicted label for test_ts: {predicted_label}")
    print("=" * 60)
    print()
    
    print("=" * 60)
    print("ANSWER TO QUESTION 3:")
    print(f"Probability associated with predicted label: {probability:.6f}")
    print("=" * 60)
    print()
    
    # Additional analysis: Show distances to nearest neighbors
    print("Additional Analysis:")
    print("-" * 60)
    distances_to_test = []
    for i, x_train in enumerate(ts_data):
        dist = dtw_dp(test_ts, x_train)
        distances_to_test.append((i, dist, y[i]))
    
    # Sort by distance
    distances_to_test.sort(key=lambda x: x[1])
    
    print(f"Top {knn_model.k} nearest neighbors to test_ts:")
    for rank, (idx, dist, label) in enumerate(distances_to_test[:knn_model.k], 1):
        weight = 1.0 / np.log(dist + 1e-8)
        print(f"  Rank {rank}: Index={idx}, Distance={dist:.6f}, Label={label}, Weight={weight:.6f}")
    print()
    
    # Summary of results
    print("=" * 60)
    print("SUMMARY OF RESULTS:")
    print("=" * 60)
    print(f"1. DTW distance (ts_data[2] to test_ts): {dist_3_test:.6f}")
    print(f"2. Predicted label for test_ts: {predicted_label}")
    print(f"3. Prediction probability: {probability:.6f}")
    print("=" * 60)
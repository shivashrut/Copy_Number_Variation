
# Online work
import matplotlib.pyplot as plt
import numpy as np
from math import *
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from scipy.stats import entropy

def modifiedEM(X, y):

    # Example usage
    # Load the dataset (genome features = RD, p, s, f, c)
    # X = [RD, p, s, f, c]
    # RD = Read Depth, p = position, s = smoothness, f = frequency content, c = correlation]
    # y = Groundtruth values

    num_clusters = 3  # Number of clusters
    min_iterations = 10  # Minimum number of iterations
    max_iterations = 200  # Maximum number of iterations
    entropy_threshold = 0.01  # Entropy threshold for convergence

    # Apply EM algorithm with supervised learning
    gmm_with_sl, num_iterations = EM_with_Supervised_Learning(X, y, num_clusters, min_iterations, max_iterations, entropy_threshold)

    # Obtain predicted labels
    predicted_labels = gmm_with_sl.predict(X)

    # Print the predicted labels
    # print(predicted_labels)
    # Print the number of iterations performed
    # print("Number of Iterations:", num_iterations)

    # Plot the clusters
    # plt.scatter(X[:, 0], X[:, 1], c=predicted_labels, cmap='viridis')
    # plt.title('Cluster Assignments')
    # plt.xlabel('Feature 1')
    # plt.ylabel('Feature 2')
    # plt.show()

    return predicted_labels

def EM_with_Supervised_Learning(X, y, num_clusters, min_iterations, max_iterations, entropy_threshold):
    # Initialize the Gaussian Mixture Model
    gmm = GaussianMixture(n_components=num_clusters)

    # Initialize the supervised learning model
    lr = LogisticRegression()

    # E-step, M-step, and supervised learning iteratively
    iteration = 0
    prev_entropy = np.inf

    while iteration < max_iterations:
        # E-step
        gmm.fit(X)  # Fit GMM to the data
        responsibilities = gmm.predict_proba(X)  # Compute fractional assignments

        # Apply fractional order assignment
        fractional_order = 20.5  # Set the fractional order
        responsibilities_raised = responsibilities ** fractional_order

        # Supervised learning step
        lr.fit(X, y)  # Fit logistic regression on the labeled data

        # M-step
        gmm.weights_ = np.sum(responsibilities_raised, axis=0) / len(X)  # Update mixing coefficients
        gmm.means_ = np.dot(responsibilities_raised.T, X) / np.sum(responsibilities_raised, axis=0)[:, np.newaxis]  # Update means

        # Update covariances
        for k in range(num_clusters):
            diff = X - gmm.means_[k]
            gmm.covariances_[k] = np.dot((responsibilities_raised[:, k] * diff.T), diff) / np.sum(responsibilities_raised[:, k])

        # Update parameters using supervised learning component
        gmm.means_ += lr.coef_  # Adjust means based on the logistic regression coefficients

        # Calculate fractional entropy
        current_entropy = entropy(responsibilities_raised.T)

        # Check convergence
        if iteration >= min_iterations and np.all(np.abs(current_entropy - prev_entropy) < entropy_threshold):
            break

        prev_entropy = current_entropy
        iteration += 1

    return gmm, iteration

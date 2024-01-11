import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load dataset
data = np.loadtxt('visualization-dataset.txt', delimiter=",")

# Create a three-dimensional scatterplot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Split dataset into two halves for different colors
half = len(data) // 2
ax.scatter(data[:half, 0], data[:half, 1], data[:half, 2], c='blue', label='First Half')
ax.scatter(data[half:, 0], data[half:, 1], data[half:, 2], c='red', label='Second Half')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title('Three-dimensional Scatterplot of Dataset')
plt.show()

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Normalize the data
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data)

# Perform PCA on unnormalized data
pca_unnormalized = PCA(n_components=2)
pca_result_unnormalized = pca_unnormalized.fit_transform(data)

# Perform PCA on normalized data
pca_normalized = PCA(n_components=2)
pca_result_normalized = pca_normalized.fit_transform(normalized_data)

# Plot results
plt.figure(figsize=(12, 5))

# Plot PCA on unnormalized data
plt.subplot(121)
plt.scatter(pca_result_unnormalized[:half, 0], pca_result_unnormalized[:half, 1], c='blue', label='First Half')
plt.scatter(pca_result_unnormalized[half:, 0], pca_result_unnormalized[half:, 1], c='red', label='Second Half')
plt.title('PCA on Unnormalized Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

# Plot PCA on normalized data
plt.subplot(122)
plt.scatter(pca_result_normalized[:half, 0], pca_result_normalized[:half, 1], c='blue', label='First Half')
plt.scatter(pca_result_normalized[half:, 0], pca_result_normalized[half:, 1], c='red', label='Second Half')
plt.title('PCA on Normalized Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

plt.tight_layout()
plt.show()

from sklearn.manifold import TSNE



# Define perplexity values
perplexities = [32, 80, 180, 280]

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('t-SNE Visualization with Different Perplexity Values')


# Apply t-SNE for each perplexity value
for i, perplexity in enumerate(perplexities):
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=300, random_state=42)
    data_tsne = tsne.fit_transform(data)

    # Plot t-SNE results
    ax = axes[i // 2, i % 2]
    ax.scatter(data_tsne[:half, 0], data_tsne[:half, 1], c='blue', label='First Half')
    ax.scatter(data_tsne[half:, 0], data_tsne[half:, 1], c='red', label='Second Half')
    ax.set_title(f't-SNE with Perplexity={perplexity}')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

normalized_data = scaler.fit_transform(data)

# Perform PCA on normalized data to get initial map points
pca_normalized = PCA(n_components=2)
initial_map_points = pca_result_normalized

# Define perplexity values
perplexities = [32, 80, 180, 280]

# Create subplots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
fig.suptitle('t-SNE Visualization with PCA Initial Map Points and Different Perplexity Values')

# Apply t-SNE for each perplexity value using PCA initial map points
for i, perplexity in enumerate(perplexities):
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=300, random_state=42, init=initial_map_points)
    data_tsne = tsne.fit_transform(normalized_data)

    # Plot t-SNE results
    ax = axes[i // 2, i % 2]
    ax.scatter(data_tsne[:half, 0], data_tsne[:half, 1], c='blue', label='First Half')
    ax.scatter(data_tsne[half:, 0], data_tsne[half:, 1], c='red', label='Second Half')
    ax.set_title(f't-SNE with PCA Initial Map Points (Perplexity={perplexity})')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import skfuzzy as fuzz
import matplotlib.pyplot as plt

npz_file = 'C:/Users/picul/Videos/Applied/Dataset_Full/Embeddings/windowed_power_embeddings.npy'
tensor = np.load(npz_file)

print(f"Tensor shape: {tensor.shape}")

sample = tensor[0]
print(f"Sample shape: {sample.shape}")

pca = PCA(n_components=2)
sample_2d = pca.fit_transform(sample)

n_clusters = 3

kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(sample)

cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
    sample.T,
    n_clusters,
    m=2,
    error=0.005,
    maxiter=1000,
    init=None,
    seed=42
)
cmeans_labels = np.argmax(u, axis=0)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

scatter1 = axes[0].scatter(sample_2d[:, 0], sample_2d[:, 1], c=kmeans_labels, cmap='viridis', s=100, alpha=0.7, edgecolors='k', linewidth=1.5)
axes[0].set_title(f'K-Means Clustering (k={n_clusters})', fontsize=13, fontweight='bold')
axes[0].set_xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=10)
axes[0].set_ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=10)
axes[0].grid(alpha=0.3)
plt.colorbar(scatter1, ax=axes[0], label='Cluster')

scatter2 = axes[1].scatter(sample_2d[:, 0], sample_2d[:, 1], c=cmeans_labels, cmap='viridis', s=100, alpha=0.7, edgecolors='k', linewidth=1.5)
axes[1].set_title(f'C-Means Clustering (k={n_clusters})', fontsize=13, fontweight='bold')
axes[1].set_xlabel(f'PCA 1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=10)
axes[1].set_ylabel(f'PCA 2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=10)
axes[1].grid(alpha=0.3)
plt.colorbar(scatter2, ax=axes[1], label='Cluster')

plt.tight_layout()
plt.savefig('clustering_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nK-Means labels: {kmeans_labels}")
print(f"C-Means labels: {cmeans_labels}")
print(f"K-Means inertia: {kmeans.inertia_:.4f}")
print(f"C-Means FPC: {fpc:.4f}")
# run PCA
from sklearn.decomposition import PCA
pca_3d = PCA(n_components=3)
pca_3d.fit(DATA_3D)

print("eigenvectors: \n", pca_3d.components_)
print("eigenvalues: \n", pca_3d.explained_variance_)
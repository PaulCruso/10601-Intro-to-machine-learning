import numpy as np

data = np.array([
    [-0.21, -0.61, -0.35,  0.08],
    [ 0.15, -0.77,  1.26,  1.57],
    [ 0.03,  0.12, -0.39, -0.25],
    [ 0.92,  1.31,  0.31,  1.19],
    [ 2.51,  1.99,  1.86,  2.57],
    [ 0.91,  1.23, -0.01,  0.04]
])

M = np.cov(data, rowvar=False)
# compute eigenvalues and eigenvectors by SVD
eigenvalues, eigenvectors = np.linalg.eigh(M)

# Sort eigenvalues/eigenvectors
sort_idx = np.argsort(eigenvalues)[::-1]
eigenvectors = eigenvectors[:, sort_idx]
eigenvalues = eigenvalues[sort_idx]

pcs = eigenvectors[:, 0, ]
reduced_representation = data @ pcs # 2 dimensional
print(reduced_representation)

reconstruction = np.outer(reduced_representation, pcs)
print(reconstruction)
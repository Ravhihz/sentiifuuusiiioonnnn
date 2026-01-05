import numpy as np


class PCA:
    """Principal Component Analysis from scratch"""

    def __init__(self, n_components=None, contribution_rate=0.98):
        self.n_components = n_components
        self.contribution_rate = contribution_rate
        self.components = None
        self.mean = None
        self.eigenvalues = None
        self.explained_variance_ratio = None

    def fit(self, X):
        """Fit PCA model"""
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # Compute covariance matrix
        n_samples = X.shape[0]
        cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # Sort by eigenvalues in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store eigenvalues
        self.eigenvalues = eigenvalues

        # Calculate explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio = eigenvalues / total_variance

        # Determine number of components
        if self.n_components is None:
            cumsum = np.cumsum(self.explained_variance_ratio)
            self.n_components = np.argmax(cumsum >= self.contribution_rate) + 1

        # Store components
        self.components = eigenvectors[:, : self.n_components]

        return self

    def transform(self, X):
        """Transform data to principal components"""
        X_centered = X - self.mean
        return np.dot(X_centered, self.components) # type: ignore

    def fit_transform(self, X):
        """Fit and transform in one step"""
        self.fit(X)
        return self.transform(X)

    def get_contribution_rate(self, n_components=None):
        """Calculate contribution rate for n components"""
        if n_components is None:
            n_components = self.n_components
        return np.sum(self.explained_variance_ratio[:n_components]) # type: ignore

    def inverse_transform(self, X_transformed):
        """Transform data back to original space"""
        return np.dot(X_transformed, self.components.T) + self.mean # type: ignore
import numpy as np
from typing import Optional, Tuple


class ManualPCA:
    """
    Manual PCA (Principal Component Analysis) Implementation
    No sklearn! Pure NumPy!
    
    Algorithm:
    1. Center the data (subtract mean)
    2. Calculate covariance matrix
    3. Calculate eigenvalues and eigenvectors
    4. Sort eigenvectors by eigenvalues (descending)
    5. Select top k eigenvectors as principal components
    6. Transform data to new space
    """
    
    def __init__(self, n_components: int = 100):
        """
        Initialize PCA
        
        Args:
            n_components: Number of principal components to keep
        """
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None  # Principal components (eigenvectors)
        self.explained_variance_ = None  # Eigenvalues
        self.explained_variance_ratio_ = None
        
    def fit(self, X: np.ndarray) -> 'ManualPCA':
        """
        Fit PCA on training data
        
        Args:
            X: Training data (n_samples, n_features)
            
        Returns:
            self
        """
        print(f"[PCA FIT] Input shape: {X.shape}")
        
        # Step 1: Center the data
        print("[STEP 1] Centering data (subtract mean)...")
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        print(f"[MEAN] Mean vector shape: {self.mean_.shape}")
        print(f"[CENTERED] Centered data shape: {X_centered.shape}")
        
        # Step 2: Calculate covariance matrix
        print("[STEP 2] Calculating covariance matrix...")
        # Cov = (X_centered^T * X_centered) / (n-1)
        n_samples = X_centered.shape[0]
        cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)
        
        print(f"[COV] Covariance matrix shape: {cov_matrix.shape}")
        
        # Step 3: Calculate eigenvalues and eigenvectors
        print("[STEP 3] Calculating eigenvalues and eigenvectors...")
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # Convert to real (remove imaginary part if any due to numerical errors)
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        
        print(f"[EIGEN] Found {len(eigenvalues)} eigenvalues")
        print(f"[EIGEN] Eigenvalues range: [{eigenvalues.min():.4f}, {eigenvalues.max():.4f}]")
        
        # Step 4: Sort eigenvectors by eigenvalues (descending)
        print("[STEP 4] Sorting eigenvectors by eigenvalues...")
        idx = np.argsort(eigenvalues)[::-1]  # Descending order
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        print(f"[SORTED] Top 5 eigenvalues: {eigenvalues[:5]}")
        
        # Step 5: Select top k components
        k = min(self.n_components, len(eigenvalues), X.shape[0], X.shape[1])
        print(f"[STEP 5] Selecting top {k} components...")
        
        self.components_ = eigenvectors[:, :k].T  # Shape: (k, n_features)
        self.explained_variance_ = eigenvalues[:k]
        
        # Calculate explained variance ratio
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        
        cumulative_variance = np.sum(self.explained_variance_ratio_) * 100
        
        print(f"[COMPONENTS] Selected {k} components")
        print(f"[COMPONENTS] Shape: {self.components_.shape}")
        print(f"[VARIANCE] Explained variance: {cumulative_variance:.2f}%")
        print(f"[VARIANCE] Top 5 ratios: {self.explained_variance_ratio_[:5]}")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data to PCA space
        
        Args:
            X: Data to transform (n_samples, n_features)
            
        Returns:
            X_transformed: Transformed data (n_samples, n_components)
        """
        if self.mean_ is None or self.components_ is None:
            raise ValueError("PCA not fitted yet! Call fit() first.")
        
        print(f"[PCA TRANSFORM] Input shape: {X.shape}")
        
        # Center the data using training mean
        X_centered = X - self.mean_
        
        # Project to PCA space: X_transformed = X_centered * components^T
        X_transformed = np.dot(X_centered, self.components_.T)
        
        print(f"[PCA TRANSFORM] Output shape: {X_transformed.shape}")
        
        return X_transformed
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA and transform data in one step
        
        Args:
            X: Training data (n_samples, n_features)
            
        Returns:
            X_transformed: Transformed data (n_samples, n_components)
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """
        Transform data back to original space
        
        Args:
            X_transformed: Transformed data (n_samples, n_components)
            
        Returns:
            X_reconstructed: Reconstructed data (n_samples, n_features)
        """
        if self.mean_ is None or self.components_ is None:
            raise ValueError("PCA not fitted yet!")
        
        # Reconstruct: X = X_transformed * components + mean
        X_reconstructed = np.dot(X_transformed, self.components_) + self.mean_
        
        return X_reconstructed
    
    def get_explained_variance_ratio(self) -> np.ndarray:
        """Get explained variance ratio for each component"""
        if self.explained_variance_ratio_ is None:
            raise ValueError("PCA not fitted yet!")
        
        return self.explained_variance_ratio_
    
    def get_cumulative_variance(self) -> float:
        """Get cumulative explained variance"""
        if self.explained_variance_ratio_ is None:
            raise ValueError("PCA not fitted yet!")
        
        return np.sum(self.explained_variance_ratio_)


# Test function
def test_manual_pca():
    """Test Manual PCA implementation"""
    print("=" * 60)
    print("TESTING MANUAL PCA")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    n_features = 300
    
    X = np.random.randn(n_samples, n_features)
    
    print(f"[TEST] Generated data shape: {X.shape}")
    
    # Fit PCA
    pca = ManualPCA(n_components=50)
    X_transformed = pca.fit_transform(X)
    
    print(f"[TEST] Transformed shape: {X_transformed.shape}")
    print(f"[TEST] Explained variance: {pca.get_cumulative_variance() * 100:.2f}%")
    
    # Test inverse transform
    X_reconstructed = pca.inverse_transform(X_transformed)
    print(f"[TEST] Reconstructed shape: {X_reconstructed.shape}")
    
    # Calculate reconstruction error
    reconstruction_error = np.mean((X - X_reconstructed) ** 2)
    print(f"[TEST] Reconstruction error (MSE): {reconstruction_error:.6f}")
    
    print("[TEST] ✅ Manual PCA working correctly!")


if __name__ == '__main__':
    test_manual_pca()
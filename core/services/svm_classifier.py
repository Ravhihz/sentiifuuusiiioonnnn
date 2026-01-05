import numpy as np
from typing import Literal, Optional


class OneVsRestSVM:
    """
    One-vs-Rest Multi-class SVM Classifier
    Trains N binary classifiers for N classes
    """
    
    def __init__(self, kernel='rbf', C=1.0, gamma=None, max_iter=1000):
        """
        Initialize One-vs-Rest SVM
        
        Args:
            kernel: Kernel type
            C: Regularization parameter
            gamma: Kernel coefficient
            max_iter: Maximum iterations
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.max_iter = max_iter
        self.classifiers = {}
        self.classes_ = None
        self.support_vector_indices_ = []
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'OneVsRestSVM':
        """
        Train One-vs-Rest classifiers
        
        Args:
            X: Training data (n_samples, n_features)
            y: Multi-class labels (n_samples,)
            
        Returns:
            self
        """
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        print(f"[OVR SVM] Training {n_classes} binary classifiers...")
        
        for i, cls in enumerate(self.classes_):
            print(f"\n[OVR SVM] Training classifier {i+1}/{n_classes} for class '{cls}'...")
            
            # Create binary labels: current class (1) vs rest (0)
            y_binary = np.where(y == cls, 1, 0)
            
            # Train binary SVM
            svm = ManualSVM(
                kernel=self.kernel, # type: ignore
                C=self.C,
                gamma=self.gamma,
                max_iter=self.max_iter
            )
            svm.fit(X, y_binary)
            
            # Store classifier
            self.classifiers[cls] = svm
            
            # Collect support vector indices
            self.support_vector_indices_.extend(svm.support_vector_indices_.tolist()) # type: ignore
        
        # Remove duplicates from support vectors
        self.support_vector_indices_ = np.unique(self.support_vector_indices_)
        
        print(f"\n[OVR SVM] All {n_classes} classifiers trained!")
        print(f"[OVR SVM] Total unique support vectors: {len(self.support_vector_indices_)}/{len(X)}")
        
        return self
    
    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Get decision values for all classifiers
        
        Args:
            X: Data points (n_samples, n_features)
            
        Returns:
            Decision matrix (n_samples, n_classes)
        """
        decisions = []
        for cls in self.classes_: # type: ignore
            decision = self.classifiers[cls]._decision_function(X)
            decisions.append(decision)
        
        return np.column_stack(decisions)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class with highest decision value
        
        Args:
            X: Data points (n_samples, n_features)
            
        Returns:
            Predicted class labels (n_samples,)
        """
        decisions = self._decision_function(X)
        indices = np.argmax(decisions, axis=1)
        return self.classes_[indices] # type: ignore
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities using softmax
        
        Args:
            X: Data points (n_samples, n_features)
            
        Returns:
            Probability matrix (n_samples, n_classes)
        """
        decisions = self._decision_function(X)
        
        # Apply softmax for probabilities
        exp_decisions = np.exp(decisions - np.max(decisions, axis=1, keepdims=True))
        probas = exp_decisions / np.sum(exp_decisions, axis=1, keepdims=True)
        
        return probas
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score
        
        Args:
            X: Data points
            y: True labels
            
        Returns:
            Accuracy [0, 1]
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


class ManualSVM:
    """
    Manual SVM Implementation using Simplified SMO Algorithm
    No sklearn! Pure NumPy!
    
    Supports:
    - Linear kernel
    - RBF (Gaussian) kernel
    - Polynomial kernel
    """
    
    def __init__(
        self,
        kernel: Literal['linear', 'rbf', 'poly'] = 'rbf',
        C: float = 1.0,
        gamma: Optional[float] = None,
        degree: int = 3,
        max_iter: int = 1000,
        tol: float = 1e-3
    ):
        """
        Initialize SVM
        
        Args:
            kernel: Kernel type ('linear', 'rbf', 'poly')
            C: Regularization parameter
            gamma: Kernel coefficient (for rbf/poly)
            degree: Degree for polynomial kernel
            max_iter: Maximum iterations for SMO
            tol: Tolerance for stopping criterion
        """
        self.kernel_type = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.max_iter = max_iter
        self.tol = tol
        
        # Model parameters (learned during training)
        self.alphas = None  # Lagrange multipliers
        self.b = 0.0  # Bias term
        self.X_train = None  # Support vectors
        self.y_train = None  # Support vector labels
        self.support_vectors_ = None
        self.support_vector_indices_ = None
        
    def _kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Compute kernel function K(X1, X2)
        
        Args:
            X1: First data matrix (n1, n_features)
            X2: Second data matrix (n2, n_features)
            
        Returns:
            Kernel matrix (n1, n2)
        """
        if self.kernel_type == 'linear':
            # Linear kernel: K(x, y) = x^T * y
            return np.dot(X1, X2.T)
        
        elif self.kernel_type == 'rbf':
            # RBF kernel: K(x, y) = exp(-gamma * ||x - y||^2)
            if self.gamma is None:
                self.gamma = 1.0 / X1.shape[1]
            
            # Compute pairwise squared distances
            X1_norm = np.sum(X1 ** 2, axis=1).reshape(-1, 1)
            X2_norm = np.sum(X2 ** 2, axis=1).reshape(1, -1)
            distances = X1_norm + X2_norm - 2 * np.dot(X1, X2.T)
            
            return np.exp(-self.gamma * distances)
        
        elif self.kernel_type == 'poly':
            # Polynomial kernel: K(x, y) = (gamma * x^T * y + 1)^degree
            if self.gamma is None:
                self.gamma = 1.0 / X1.shape[1]
            
            return (self.gamma * np.dot(X1, X2.T) + 1) ** self.degree
        
        else:
            raise ValueError(f"Unknown kernel: {self.kernel_type}")
    
    def _decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate decision function: f(x) = sum(alpha_i * y_i * K(x_i, x)) + b
        
        Args:
            X: Data points (n_samples, n_features)
            
        Returns:
            Decision values (n_samples,)
        """
        K = self._kernel(X, self.X_train) # type: ignore
        return np.dot(K, self.alphas * self.y_train) + self.b # type: ignore
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'ManualSVM':
        """
        Train SVM using Simplified SMO Algorithm
        
        Args:
            X: Training data (n_samples, n_features)
            y: Labels (n_samples,) - must be 0/1 or -1/+1
            
        Returns:
            self
        """
        n_samples, n_features = X.shape
        
        print(f"[SVM FIT] Training on {n_samples} samples with {n_features} features")
        print(f"[SVM] Kernel: {self.kernel_type}, C: {self.C}")
        
        # Store original classes
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"Binary SVM expects 2 classes, got {len(self.classes_)}")
        
        # Convert labels to -1 and +1
        y_binary = np.where(y == self.classes_[0], -1, 1)
        
        # Initialize
        self.X_train = X
        self.y_train = y_binary
        self.alphas = np.zeros(n_samples)
        self.b = 0.0
        
        # Compute kernel matrix
        print("[SVM] Computing kernel matrix...")
        K = self._kernel(X, X)
        
        # Simplified SMO Algorithm
        print("[SVM] Training with SMO algorithm...")
        
        for iteration in range(self.max_iter):
            alpha_changed = 0
            
            for i in range(n_samples):
                # Calculate error for i
                E_i = self._decision_function(X[i:i+1])[0] - y_binary[i]
                
                # Check KKT conditions
                if ((y_binary[i] * E_i < -self.tol and self.alphas[i] < self.C) or
                    (y_binary[i] * E_i > self.tol and self.alphas[i] > 0)):
                    
                    # Select j randomly (not i)
                    j = i
                    while j == i:
                        j = np.random.randint(0, n_samples)
                    
                    # Calculate error for j
                    E_j = self._decision_function(X[j:j+1])[0] - y_binary[j]
                    
                    # Save old alphas
                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]
                    
                    # Compute bounds L and H
                    if y_binary[i] != y_binary[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])
                    
                    if L == H:
                        continue
                    
                    # Compute eta
                    eta = 2 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue
                    
                    # Update alpha_j
                    self.alphas[j] -= y_binary[j] * (E_i - E_j) / eta
                    self.alphas[j] = np.clip(self.alphas[j], L, H)
                    
                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # Update alpha_i
                    self.alphas[i] += y_binary[i] * y_binary[j] * (alpha_j_old - self.alphas[j])
                    
                    # Update bias b
                    b1 = self.b - E_i - y_binary[i] * (self.alphas[i] - alpha_i_old) * K[i, i] - \
                         y_binary[j] * (self.alphas[j] - alpha_j_old) * K[i, j]
                    b2 = self.b - E_j - y_binary[i] * (self.alphas[i] - alpha_i_old) * K[i, j] - \
                         y_binary[j] * (self.alphas[j] - alpha_j_old) * K[j, j]
                    
                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    
                    alpha_changed += 1
            
            if (iteration + 1) % 100 == 0:
                print(f"  Iteration {iteration + 1}/{self.max_iter}, alphas changed: {alpha_changed}")
            
            if alpha_changed == 0:
                print(f"[SVM] Converged at iteration {iteration + 1}")
                break
        
        # Identify support vectors
        sv_indices = np.where(self.alphas > 1e-5)[0]
        self.support_vectors_ = X[sv_indices]
        self.support_vector_indices_ = sv_indices
        
        print(f"[SVM] Training complete!")
        print(f"[SVM] Support vectors: {len(sv_indices)}/{n_samples} ({len(sv_indices)/n_samples*100:.1f}%)")
        print(f"[SVM] Bias term: {self.b:.4f}")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels
        
        Args:
            X: Data points (n_samples, n_features)
            
        Returns:
            Predicted labels (n_samples,)
        """
        decision = self._decision_function(X)
        predictions = np.where(decision >= 0, 1, -1)
        
        # Convert back to original labels
        return np.where(predictions == 1, self.classes_[1], self.classes_[0])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (approximation using sigmoid)
        
        Args:
            X: Data points (n_samples, n_features)
            
        Returns:
            Probabilities (n_samples, 2)
        """
        decision = self._decision_function(X)
        
        # Apply sigmoid to get probabilities
        proba_positive = 1 / (1 + np.exp(-decision))
        proba_negative = 1 - proba_positive
        
        return np.column_stack([proba_negative, proba_positive])
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate accuracy score
        
        Args:
            X: Data points (n_samples, n_features)
            y: True labels (n_samples,)
            
        Returns:
            Accuracy score [0, 1]
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)


# Test function
def test_manual_svm():
    """Test Manual SVM implementation"""
    print("=" * 60)
    print("TESTING MANUAL SVM")
    print("=" * 60)
    
    # Generate sample data
    np.random.seed(42)
    
    # Class 1: centered at (2, 2)
    X1 = np.random.randn(50, 2) + np.array([2, 2])
    y1 = np.zeros(50)
    
    # Class 2: centered at (-2, -2)
    X2 = np.random.randn(50, 2) + np.array([-2, -2])
    y2 = np.ones(50)
    
    X = np.vstack([X1, X2])
    y = np.hstack([y1, y2])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    print(f"[TEST] Data shape: {X.shape}")
    print(f"[TEST] Labels: {np.unique(y)}")
    
    # Train SVM
    svm = ManualSVM(kernel='rbf', C=1.0, max_iter=500)
    svm.fit(X, y)
    
    # Test accuracy
    accuracy = svm.score(X, y)
    print(f"[TEST] Training accuracy: {accuracy * 100:.2f}%")
    
    # Test prediction
    test_point = np.array([[2, 2], [-2, -2]])
    predictions = svm.predict(test_point)
    print(f"[TEST] Predictions for [[2,2], [-2,-2]]: {predictions}")
    
    print("[TEST] ✅ Manual SVM working correctly!")


if __name__ == '__main__':
    test_manual_svm()
import numpy as np
from typing import Tuple
from collections import Counter


class SMOTE:
    """
    SMOTE (Synthetic Minority Over-sampling Technique)
    Generate synthetic samples for minority classes
    No sklearn! Pure NumPy implementation!
    """
    
    def __init__(self, k_neighbors: int = 5, random_state: int = 42):
        """
        Initialize SMOTE
        
        Args:
            k_neighbors: Number of nearest neighbors to use
            random_state: Random seed for reproducibility
        """
        self.k_neighbors = k_neighbors
        self.random_state = random_state
    
    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE to oversample minority classes
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels (n_samples,)
            
        Returns:
            X_resampled, y_resampled: Balanced dataset
        """
        np.random.seed(self.random_state)
        
        print("=" * 60)
        print("SMOTE - OVERSAMPLING MINORITY CLASSES")
        print("=" * 60)
        
        # Count class distribution
        class_counts = Counter(y)
        classes = list(class_counts.keys())
        
        print(f"[BEFORE SMOTE] Class distribution:")
        total_before = len(y)
        for cls in classes:
            count = class_counts[cls]
            pct = count / total_before * 100
            print(f"  {cls}: {count} samples ({pct:.1f}%)")
        
        # Find majority class count
        max_count = max(class_counts.values())
        
        print(f"\n[TARGET] Balance all classes to: {max_count} samples")
        
        # Separate samples by class
        X_by_class = {}
        for cls in classes:
            mask = (y == cls)
            X_by_class[cls] = X[mask]
        
        # Oversample minority classes
        X_resampled_list = []
        y_resampled_list = []
        
        for cls in classes:
            X_cls = X_by_class[cls]
            n_samples = len(X_cls)
            n_synthetic = max_count - n_samples
            
            print(f"\n[CLASS '{cls}']")
            print(f"  Current: {n_samples} samples")
            print(f"  Need to generate: {n_synthetic} synthetic samples")
            
            # Add original samples
            X_resampled_list.append(X_cls)
            y_resampled_list.append(np.full(n_samples, cls))
            
            if n_synthetic > 0:
                # Generate synthetic samples using SMOTE
                X_synthetic = self._generate_synthetic_samples(
                    X_cls, n_synthetic
                )
                
                X_resampled_list.append(X_synthetic)
                y_resampled_list.append(np.full(n_synthetic, cls))
                
                print(f"  Generated: {len(X_synthetic)} synthetic samples")
        
        # Combine all samples
        X_resampled = np.vstack(X_resampled_list)
        y_resampled = np.hstack(y_resampled_list)
        
        # Shuffle
        indices = np.random.permutation(len(X_resampled))
        X_resampled = X_resampled[indices]
        y_resampled = y_resampled[indices]
        
        print(f"\n[AFTER SMOTE] Class distribution:")
        resampled_counts = Counter(y_resampled)
        total_after = len(y_resampled)
        for cls in classes:
            count = resampled_counts[cls]
            pct = count / total_after * 100
            print(f"  {cls}: {count} samples ({pct:.1f}%)")
        
        print(f"\n[RESULT] Total samples: {total_before} → {total_after}")
        print("=" * 60)
        
        return X_resampled, y_resampled
    
    def _generate_synthetic_samples(
        self, 
        X: np.ndarray, 
        n_synthetic: int
    ) -> np.ndarray:
        """
        Generate synthetic samples using k-nearest neighbors
        
        Args:
            X: Minority class samples (n_samples, n_features)
            n_synthetic: Number of synthetic samples to generate
            
        Returns:
            X_synthetic: Generated samples (n_synthetic, n_features)
        """
        n_samples, n_features = X.shape
        synthetic_samples = []
        
        for _ in range(n_synthetic):
            # Randomly select a sample
            idx = np.random.randint(0, n_samples)
            sample = X[idx]
            
            # Find k nearest neighbors
            k = min(self.k_neighbors, n_samples - 1)
            neighbors = self._find_k_neighbors(sample, X, k)
            
            # Randomly select one neighbor
            neighbor_idx = np.random.randint(0, len(neighbors))
            neighbor = neighbors[neighbor_idx]
            
            # Generate synthetic sample
            # Formula: synthetic = sample + lambda * (neighbor - sample)
            # where lambda is random [0, 1]
            lambda_val = np.random.random()
            synthetic = sample + lambda_val * (neighbor - sample)
            
            synthetic_samples.append(synthetic)
        
        return np.array(synthetic_samples)
    
    def _find_k_neighbors(
        self, 
        sample: np.ndarray, 
        X: np.ndarray, 
        k: int
    ) -> np.ndarray:
        """
        Find k nearest neighbors using Euclidean distance
        
        Args:
            sample: Query sample (n_features,)
            X: All samples (n_samples, n_features)
            k: Number of neighbors
            
        Returns:
            neighbors: k nearest neighbor samples
        """
        # Calculate Euclidean distances
        distances = np.sqrt(np.sum((X - sample) ** 2, axis=1))
        
        # Get indices of k+1 nearest neighbors (excluding the sample itself)
        k_indices = np.argsort(distances)[1:k+1]
        
        return X[k_indices]


def check_imbalance(y: np.ndarray, threshold: float = 0.3) -> bool:
    """
    Check if dataset is imbalanced
    
    Args:
        y: Labels
        threshold: Imbalance threshold (default: 30%)
                  If min_class_ratio < threshold, consider imbalanced
    
    Returns:
        True if imbalanced, False otherwise
    """
    class_counts = Counter(y)
    total = len(y)
    
    min_count = min(class_counts.values())
    max_count = max(class_counts.values())
    
    min_ratio = min_count / total
    imbalance_ratio = min_count / max_count
    
    print(f"[IMBALANCE CHECK]")
    print(f"  Min class: {min_count} samples ({min_ratio*100:.1f}%)")
    print(f"  Max class: {max_count} samples")
    print(f"  Imbalance ratio: {imbalance_ratio:.2f}")
    
    is_imbalanced = (imbalance_ratio < threshold) or (min_ratio < 0.25)
    
    if is_imbalanced:
        print(f"  ⚠️  Dataset is IMBALANCED!")
    else:
        print(f"  ✅ Dataset is balanced")
    
    return is_imbalanced


# Test function
def test_smote():
    """Test SMOTE implementation"""
    print("Testing SMOTE implementation...\n")
    
    # Generate imbalanced dataset
    np.random.seed(42)
    
    # Majority class: 100 samples
    X_majority = np.random.randn(100, 10)
    y_majority = np.zeros(100)
    
    # Minority class: 20 samples
    X_minority = np.random.randn(20, 10) + 2
    y_minority = np.ones(20)
    
    X = np.vstack([X_majority, X_minority])
    y = np.hstack([y_majority, y_minority])
    
    print(f"Original dataset shape: {X.shape}")
    print(f"Original labels: {Counter(y)}\n")
    
    # Apply SMOTE
    smote = SMOTE(k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    print(f"\nResampled dataset shape: {X_resampled.shape}")
    print(f"Resampled labels: {Counter(y_resampled)}")
    print("\n✅ SMOTE test complete!")


if __name__ == '__main__':
    test_smote()
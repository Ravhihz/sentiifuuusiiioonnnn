import numpy as np
import json
import pickle
import os
from typing import Dict, Tuple, List, Optional
from django.conf import settings

from core.services.lda_topic_model import LDATopicModel
from core.services.smote import SMOTE, check_imbalance

from ..models import Dataset, Review, ProcessedReview, SentimentModel
from .pca_reducer import ManualPCA
from .svm_classifier import ManualSVM


class TrainingService:
    """
    Training Service - Orchestrates the entire ML pipeline
    
    Pipeline:
    1. Load labeled data (feature vectors)
    2. Train-test split (stratified)
    3. Balance data with SMOTE (if imbalanced)
    4. PCA dimensionality reduction (300 → n_components)
    5. SVM training (RBF kernel)
    6. Evaluation (accuracy, precision, recall, F1)
    7. LDA topic extraction (optional)
    8. Save model to disk
    """
    
    def __init__(self):
        pass
    
    def prepare_data(self, dataset: Dataset) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """
        Load and prepare training data from labeled reviews
        
        Args:
            dataset: Dataset object
            
        Returns:
            X: Feature vectors (n_samples, 300)
            y: Labels as integers (n_samples,)
            review_ids: List of review IDs
        """
        print("[STEP 1] Preparing training data...")
        
        # Get labeled reviews that have been preprocessed
        processed_reviews = ProcessedReview.objects.filter(
            review__dataset=dataset,
            review__label__isnull=False
        ).select_related('review')
        
        if processed_reviews.count() == 0:
            raise ValueError("No labeled reviews found! Please label some reviews first.")
        
        X = []
        y = []
        review_ids = []
        
        for processed in processed_reviews:
            # Get feature vector (300-dim from FastText)
            features = json.loads(processed.feature_vector)
            X.append(features)
            
            # Get label
            label = processed.review.label
            y.append(label)
            
            review_ids.append(processed.review.id) # type: ignore
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"[DATA] Loaded {len(X)} labeled samples")
        print(f"[DATA] Feature shape: {X.shape}")
        print(f"[DATA] Labels: {np.unique(y)}")
        
        # Count class distribution
        unique, counts = np.unique(y, return_counts=True)
        class_dist = dict(zip(unique, counts))
        print(f"[DATA] Class distribution: {class_dist}")
        
        return X, y, review_ids
    
    def train_test_split(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and test sets (stratified)
        
        Args:
            X: Features
            y: Labels
            test_size: Ratio of test set
            random_state: Random seed
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        print(f"[STEP 2] Splitting data (test_size={test_size})...")
        
        np.random.seed(random_state)
        
        # Stratified split (maintain class distribution)
        classes = np.unique(y)
        train_indices = []
        test_indices = []
        
        for cls in classes:
            cls_indices = np.where(y == cls)[0]
            np.random.shuffle(cls_indices)
            
            n_test = int(len(cls_indices) * test_size)
            test_indices.extend(cls_indices[:n_test])
            train_indices.extend(cls_indices[n_test:])
        
        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)
        
        X_train = X[train_indices]
        X_test = X[test_indices]
        y_train = y[train_indices]
        y_test = y[test_indices]
        
        print(f"[SPLIT] Train: {len(X_train)} samples")
        print(f"[SPLIT] Test: {len(X_test)} samples")
        print(f"[SPLIT] Train distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
        print(f"[SPLIT] Test distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")
        
        return X_train, X_test, y_train, y_test
    
    def balance_data(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        apply_smote: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Check imbalance and apply SMOTE if needed
        
        Args:
            X_train: Training features
            y_train: Training labels
            apply_smote: Whether to apply SMOTE (default: True)
            
        Returns:
            X_balanced, y_balanced
        """
        print("[STEP 2.5] Checking class balance...")
        
        # Check if imbalanced
        is_imbalanced = check_imbalance(y_train, threshold=0.3)
        
        if is_imbalanced and apply_smote:
            print("[SMOTE] Applying SMOTE to balance classes...")
            smote = SMOTE(k_neighbors=5)
            X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
            return X_balanced, y_balanced
        else:
            if not apply_smote:
                print("[SKIP] SMOTE disabled by user")
            else:
                print("[SKIP] Dataset is already balanced")
            return X_train, y_train
    
    def apply_pca(
        self, 
        X_train: np.ndarray, 
        X_test: np.ndarray,
        n_components: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, ManualPCA]:
        """
        Apply PCA dimensionality reduction
        
        Args:
            X_train: Training features
            X_test: Test features
            n_components: Number of components to keep
            
        Returns:
            X_train_pca, X_test_pca, pca_model
        """
        print(f"[STEP 3] Applying PCA (n_components={n_components})...")
        
        pca = ManualPCA(n_components=n_components)
        X_train_pca = pca.fit_transform(X_train)
        X_test_pca = pca.transform(X_test)
        
        explained_variance = pca.get_cumulative_variance() * 100
        
        print(f"[PCA] Reduced: {X_train.shape[1]} → {X_train_pca.shape[1]} dimensions")
        print(f"[PCA] Explained variance: {explained_variance:.2f}%")
        
        return X_train_pca, X_test_pca, pca
    
    def train_svm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        kernel: str = 'rbf',
        C: float = 1.0,
        max_iter: int = 1000
    ):
        """
        Train SVM classifier (auto-detect binary vs multi-class)
        """
        from .svm_classifier import OneVsRestSVM
        
        n_classes = len(np.unique(y_train))
        
        print(f"[STEP 4] Training SVM (kernel={kernel}, C={C})...")
        print(f"[SVM] Detected {n_classes} classes")
        
        if n_classes > 2:
            print("[SVM] Multi-class detected! Using One-vs-Rest strategy...")
            svm = OneVsRestSVM(kernel=kernel, C=C, max_iter=max_iter)
        else:
            print("[SVM] Binary classification")
            from .svm_classifier import ManualSVM
            svm = ManualSVM(kernel=kernel, C=C, max_iter=max_iter) # type: ignore
        
        svm.fit(X_train, y_train)
        
        return svm
    
    def evaluate_model(
        self,
        svm,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Dict:
        """
        Evaluate model performance
        
        Args:
            svm: Trained SVM model
            X_train, y_train: Training data
            X_test, y_test: Test data
            
        Returns:
            Dictionary with metrics
        """
        print("[STEP 5] Evaluating model...")
        
        # Predictions
        y_train_pred = svm.predict(X_train)
        y_test_pred = svm.predict(X_test)
        
        # Calculate metrics
        train_accuracy = np.mean(y_train_pred == y_train)
        test_accuracy = np.mean(y_test_pred == y_test)
        
        # Per-class metrics (weighted for multi-class)
        classes = np.unique(y_test)
        
        # Precision, Recall, F1 per class
        precision_per_class = {}
        recall_per_class = {}
        f1_per_class = {}
        
        for cls in classes:
            # True Positives, False Positives, False Negatives
            tp = np.sum((y_test_pred == cls) & (y_test == cls))
            fp = np.sum((y_test_pred == cls) & (y_test != cls))
            fn = np.sum((y_test_pred != cls) & (y_test == cls))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            precision_per_class[cls] = precision
            recall_per_class[cls] = recall
            f1_per_class[cls] = f1
        
        # Weighted average (by support)
        class_counts = {cls: np.sum(y_test == cls) for cls in classes}
        total = len(y_test)
        
        precision_weighted = sum(precision_per_class[cls] * class_counts[cls] / total for cls in classes)
        recall_weighted = sum(recall_per_class[cls] * class_counts[cls] / total for cls in classes)
        f1_weighted = sum(f1_per_class[cls] * class_counts[cls] / total for cls in classes)
        
        print(f"[METRICS] Train Accuracy: {train_accuracy*100:.2f}%")
        print(f"[METRICS] Test Accuracy: {test_accuracy*100:.2f}%")
        print(f"[METRICS] Precision (weighted): {precision_weighted:.4f}")
        print(f"[METRICS] Recall (weighted): {recall_weighted:.4f}")
        print(f"[METRICS] F1-Score (weighted): {f1_weighted:.4f}")
        
        # Per-class metrics
        print("[METRICS] Per-class metrics:")
        for cls in classes:
            print(f"  {cls}: P={precision_per_class[cls]:.3f}, R={recall_per_class[cls]:.3f}, F1={f1_per_class[cls]:.3f}")
        
        return {
            'train_accuracy': float(train_accuracy),
            'test_accuracy': float(test_accuracy),
            'precision': float(precision_weighted),
            'recall': float(recall_weighted),
            'f1_score': float(f1_weighted),
            'precision_per_class': {str(k): float(v) for k, v in precision_per_class.items()},
            'recall_per_class': {str(k): float(v) for k, v in recall_per_class.items()},
            'f1_per_class': {str(k): float(v) for k, v in f1_per_class.items()},
            'class_counts': {str(k): int(v) for k, v in class_counts.items()},
            'y_test': [str(x) for x in y_test.tolist()],
            'y_test_pred': [str(x) for x in y_test_pred.tolist()]
        }
    
    def extract_topics(
        self,
        dataset: Dataset,
        n_topics: int = 5,
        n_iterations: int = 100
    ) -> Dict:
        """
        Extract topics using LDA
        
        Args:
            dataset: Dataset object
            n_topics: Number of topics to extract
            n_iterations: LDA iterations
            
        Returns:
            Dictionary with topics
        """
        print(f"[STEP 6] Extracting topics with LDA (n_topics={n_topics})...")
        
        # Get tokenized texts from processed reviews
        processed_reviews = ProcessedReview.objects.filter(
            review__dataset=dataset
        )
        
        documents = []
        for processed in processed_reviews:
            tokens = json.loads(processed.tokenized_text)
            documents.append(tokens)
        
        print(f"[LDA] Processing {len(documents)} documents...")
        
        # Train LDA
        lda = LDATopicModel(
            n_topics=n_topics,
            n_iterations=n_iterations,
            alpha=0.1,
            beta=0.01
        )
        lda.fit(documents)
        
        # Get topics
        topics = lda.get_topics()
        
        # Convert to serializable format
        topics_dict = {}
        for topic_idx, words in enumerate(topics):
            topics_dict[f"topic_{topic_idx + 1}"] = [
                {"word": word, "probability": float(prob)}
                for word, prob in words
            ]
        
        # Save topics to file
        topics_dir = os.path.join(settings.MEDIA_ROOT, 'topics')
        os.makedirs(topics_dir, exist_ok=True)
        
        import time
        timestamp = int(time.time())
        topics_filename = f'topics_{dataset.pk}_{timestamp}.json'
        topics_path = os.path.join(topics_dir, topics_filename)
        
        lda.save_topics(topics_path)
        
        print(f"[LDA] Topics extracted successfully!")
        
        return {
            'topics': topics_dict,
            'n_topics': n_topics,
            'topics_file': f'topics/{topics_filename}'
        }
    
    def save_model(
        self,
        dataset: Dataset,
        pca: ManualPCA,
        svm,
        metrics: Dict,
        parameters: Dict,
        topics_info: Optional[Dict] = None
    ) -> SentimentModel:
        """
        Save trained model to disk and database
        
        Args:
            dataset: Dataset object
            pca: Trained PCA model
            svm: Trained SVM model
            metrics: Evaluation metrics
            parameters: Training parameters
            topics_info: LDA topics (optional)
            
        Returns:
            SentimentModel database record
        """
        print("[STEP 7] Saving model...")
        
        # Create models directory
        models_dir = os.path.join(settings.MEDIA_ROOT, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Generate filename
        import time
        timestamp = int(time.time())
        model_filename = f'svm_model_{dataset.pk}_{timestamp}.pkl'
        model_path = os.path.join(models_dir, model_filename)
        
        # Save model components
        model_data = {
            'pca': pca,
            'svm': svm,
            'classes': svm.classes_,
            'parameters': parameters
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"[SAVE] Model saved to: {model_path}")
        
        # Add topics to parameters if available
        if topics_info:
            parameters['topics'] = topics_info['topics']
            parameters['n_topics'] = topics_info['n_topics']
            parameters['topics_file'] = topics_info['topics_file']
        
        # Create database record
        model_record = SentimentModel.objects.create(
            dataset=dataset,
            name=f"SVM-PCA Model ({dataset.name})",
            algorithm='svm',
            parameters=parameters,
            accuracy=metrics['test_accuracy'],
            precision=metrics['precision'],
            recall=metrics['recall'],
            f1_score=metrics['f1_score'],
            model_file=f'models/{model_filename}'
        )
        
        print(f"[DATABASE] Model record created with ID: {model_record.pk}")
        
        return model_record
    
    def train_model(
        self,
        dataset: Dataset,
        test_size: float = 0.2,
        n_components: int = 100,
        kernel: str = 'rbf',
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
        apply_smote: bool = True,
        extract_lda_topics: bool = True,
        n_topics: int = 5
    ) -> Dict:
        """
        Complete training pipeline with SMOTE + LDA
        
        Args:
            dataset: Dataset object
            test_size: Test split ratio
            n_components: PCA components
            kernel: SVM kernel type
            C: SVM regularization
            max_iter: Max SVM iterations
            random_state: Random seed
            apply_smote: Whether to apply SMOTE
            extract_lda_topics: Whether to extract topics
            n_topics: Number of LDA topics
            
        Returns:
            Training results dictionary
        """
        print("=" * 60)
        print("TRAINING PIPELINE - SVM + PCA + SMOTE + LDA")
        print("=" * 60)
        
        import time
        start_time = time.time()
        
        # Step 1: Prepare data
        X, y, review_ids = self.prepare_data(dataset)
        
        # Step 2: Train-test split
        X_train, X_test, y_train, y_test = self.train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Step 2.5: Balance data with SMOTE
        X_train_balanced, y_train_balanced = self.balance_data(
            X_train, y_train, apply_smote=apply_smote
        )
        
        # Step 3: PCA
        X_train_pca, X_test_pca, pca = self.apply_pca(
            X_train_balanced, X_test, n_components=n_components
        )
        
        # Step 4: Train SVM
        svm = self.train_svm(
            X_train_pca, y_train_balanced, kernel=kernel, C=C, max_iter=max_iter
        )
        
        # Step 5: Evaluate
        metrics = self.evaluate_model(
            svm, X_train_pca, y_train_balanced, X_test_pca, y_test
        )
        
        # Step 6: Extract topics with LDA (optional)
        topics_info = None
        if extract_lda_topics:
            topics_info = self.extract_topics(
                dataset, n_topics=n_topics, n_iterations=100
            )
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Step 7: Prepare parameters
        parameters = {
            'test_size': test_size,
            'train_samples': int(len(X_train_balanced)),
            'test_samples': int(len(X_test)),
            'n_components': n_components,
            'pca_variance': float(pca.get_cumulative_variance()),
            'kernel': kernel,
            'C': C,
            'max_iter': max_iter,
            'random_state': random_state,
            'apply_smote': apply_smote,
            'n_samples_before_smote': int(len(X_train)),
            'n_samples_after_smote': int(len(X_train_balanced)),
            'n_features_original': int(X.shape[1]),
            'n_support_vectors': len(svm.support_vector_indices_), # type: ignore
            'support_vector_ratio': float(len(svm.support_vector_indices_) / len(X_train_balanced)), # type: ignore
            'training_time': float(total_time),
            'train_accuracy': float(metrics['train_accuracy']),
            'extract_lda_topics': extract_lda_topics,
            'n_topics': n_topics if extract_lda_topics else 0
        }
        
        # Step 8: Save model
        model_record = self.save_model(dataset, pca, svm, metrics, parameters, topics_info)
        
        print("=" * 60)
        print(f"TRAINING COMPLETE! Total time: {total_time:.2f}s")
        print("=" * 60)
        
        return {
            'model_id': int(model_record.pk),
            'train_samples': int(len(X_train_balanced)),
            'test_samples': int(len(X_test)),
            'n_features_original': int(X.shape[1]),
            'n_features_pca': int(X_train_pca.shape[1]),
            'pca_variance': float(pca.get_cumulative_variance()),
            'n_support_vectors': int(len(svm.support_vector_indices_)), # type: ignore
            'training_time': float(total_time),
            'topics': topics_info['topics'] if topics_info else None,
            **metrics,
            'parameters': parameters
        }
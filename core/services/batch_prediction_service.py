import numpy as np
import json
import pickle
import os
from typing import Dict, List
from django.conf import settings
from django.utils import timezone
from django.db import models

from ..models import Dataset, Review, ProcessedReview, SentimentModel


class BatchPredictionService:
    """
    Batch Prediction Service
    Apply trained model to all unlabeled reviews in dataset
    """
    
    def __init__(self):
        pass
    
    def load_model(self, model: SentimentModel) -> Dict:
        """
        Load trained model from disk
        
        Args:
            model: SentimentModel instance
            
        Returns:
            Dictionary with model components
        """
        print(f"[LOAD] Loading model: {model.name}")
        
        model_path = os.path.join(settings.MEDIA_ROOT, model.model_file)
        
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        
        print(f"[LOAD] Model loaded successfully")
        
        return model_data
    
    def predict_batch(
        self,
        dataset: Dataset,
        model: SentimentModel,
        overwrite_existing: bool = False
    ) -> Dict:
        """
        Apply model to all unlabeled reviews
        
        Args:
            dataset: Dataset to predict
            model: Trained model to use
            overwrite_existing: Whether to overwrite existing predictions
            
        Returns:
            Prediction results summary
        """
        print("=" * 60)
        print("BATCH PREDICTION")
        print("=" * 60)
        
        import time
        start_time = time.time()
        
        # Load model
        model_data = self.load_model(model)
        pca = model_data['pca']
        svm = model_data['svm']
        
        # Get unlabeled reviews
        if overwrite_existing:
            # Predict all reviews (including previously predicted)
            reviews_to_predict = Review.objects.filter(
                dataset=dataset,
                label__isnull=True  # Only unlabeled (manual label)
            )
        else:
            # Only predict new ones (never predicted before)
            reviews_to_predict = Review.objects.filter(
                dataset=dataset,
                label__isnull=True,
                predicted_label__isnull=True
            )
        
        total_reviews = reviews_to_predict.count()
        
        if total_reviews == 0:
            print("[WARNING] No reviews to predict!")
            return {
                'status': 'no_data',
                'message': 'No unlabeled reviews found',
                'total_reviews': 0
            }
        
        print(f"[BATCH] Predicting {total_reviews} reviews...")
        
        # Prepare data
        X = []
        review_ids = []
        
        for review in reviews_to_predict:
            try:
                # Get processed review
                processed = ProcessedReview.objects.get(review=review)
                
                # Get feature vector
                features = json.loads(processed.feature_vector)
                X.append(features)
                review_ids.append(review.id) # type: ignore
                
            except ProcessedReview.DoesNotExist:
                print(f"[SKIP] Review {review.id} not preprocessed") # type: ignore
                continue
        
        if len(X) == 0:
            print("[ERROR] No valid feature vectors found!")
            return {
                'status': 'error',
                'message': 'No preprocessed reviews found',
                'total_reviews': 0
            }
        
        X = np.array(X)
        print(f"[DATA] Prepared {len(X)} feature vectors")
        
        # Apply PCA
        print("[PCA] Applying dimensionality reduction...")
        X_pca = pca.transform(X)
        
        # Predict
        print("[SVM] Predicting sentiments...")
        predictions = svm.predict(X_pca)
        probabilities = svm.predict_proba(X_pca)
        
        # Get confidence scores (max probability)
        confidences = np.max(probabilities, axis=1)
        
        # Save predictions
        print("[SAVE] Saving predictions to database...")
        
        prediction_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        
        for i, review_id in enumerate(review_ids):
            review = Review.objects.get(id=review_id)
            
            predicted_sentiment = predictions[i]
            confidence = float(confidences[i])
            
            # Update review
            review.predicted_label = predicted_sentiment
            review.prediction_confidence = confidence
            review.predicted_by_model = model # type: ignore
            review.predicted_at = timezone.now()
            review.save()
            
            # Count
            prediction_counts[predicted_sentiment] += 1
            
            if (i + 1) % 100 == 0:
                print(f"  Saved {i + 1}/{len(review_ids)} predictions...")
        
        total_time = time.time() - start_time
        
        print("=" * 60)
        print(f"BATCH PREDICTION COMPLETE! Time: {total_time:.2f}s")
        print("=" * 60)
        print(f"[RESULTS] Prediction Distribution:")
        for sentiment, count in prediction_counts.items():
            percentage = count / len(review_ids) * 100
            print(f"  {sentiment}: {count} ({percentage:.1f}%)")
        
        return {
            'status': 'completed',
            'total_reviews': len(review_ids),
            'predictions': prediction_counts,
            'processing_time': total_time,
            'model_id': model.id, # type: ignore
            'model_name': model.name,
            'dataset_id': dataset.id, # type: ignore
            'dataset_name': dataset.name
        }
    
    def get_prediction_summary(self, dataset: Dataset) -> Dict:
        """
        Get summary of predictions for dataset
        
        Args:
            dataset: Dataset to summarize
            
        Returns:
            Summary statistics
        """
        total_reviews = dataset.reviews.count() # type: ignore
        
        # Manual labels
        labeled_reviews = dataset.reviews.filter(label__isnull=False) # type: ignore
        labeled_count = labeled_reviews.count()
        
        label_distribution = {}
        for label in ['positive', 'negative', 'neutral']:
            count = labeled_reviews.filter(label=label).count()
            label_distribution[label] = count
        
        # Predictions
        predicted_reviews = dataset.reviews.filter(predicted_label__isnull=False) # type: ignore
        predicted_count = predicted_reviews.count()
        
        prediction_distribution = {}
        for label in ['positive', 'negative', 'neutral']:
            count = predicted_reviews.filter(predicted_label=label).count()
            prediction_distribution[label] = count
        
        # Unlabeled (no manual label, no prediction)
        unlabeled_count = dataset.reviews.filter( # type: ignore
            label__isnull=True,
            predicted_label__isnull=True
        ).count()
        
        # Average confidence
        avg_confidence = predicted_reviews.aggregate(
            avg_conf=models.Avg('prediction_confidence')
        )['avg_conf']
        
        return {
            'total_reviews': total_reviews,
            'labeled_count': labeled_count,
            'predicted_count': predicted_count,
            'unlabeled_count': unlabeled_count,
            'label_distribution': label_distribution,
            'prediction_distribution': prediction_distribution,
            'avg_confidence': avg_confidence or 0.0
        }
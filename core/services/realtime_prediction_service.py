import numpy as np
import json
import pickle
import os
from typing import Dict, Tuple
from django.conf import settings

from ..models import SentimentModel, PredictionRequest
from .text_preprocessor import TextPreprocessor
from .feature_extractor import FeatureExtractor


class RealtimePredictionService:
    """
    Real-time Prediction Service
    Predict sentiment for new text input (production data)
    """
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.feature_extractor = None
    
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
    
    def preprocess_text(self, text: str) -> Tuple[str, list]:
        """
        Preprocess input text
        
        Args:
            text: Raw input text
            
        Returns:
            cleaned_text, tokens
        """
        print(f"[PREPROCESS] Cleaning text...")
        
        # Clean text
        cleaned = self.preprocessor.clean_text(text)
        
        # Tokenize
        tokens = self.preprocessor.tokenize(cleaned)
        
        print(f"[PREPROCESS] Tokens: {len(tokens)}")
        
        return cleaned, tokens
    
    def extract_features(self, tokens: list) -> np.ndarray:
        """
        Extract FastText feature vector
        
        Args:
            tokens: List of tokens
            
        Returns:
            Feature vector (300-dim)
        """
        print(f"[FEATURES] Extracting FastText features...")
        
        # Load FastText model if not loaded
        if self.feature_extractor is None:
            fasttext_path = os.path.join(
                settings.BASE_DIR, 
                'core', 'ml_models', 'cc.id.300.bin'
            )
            
            from gensim.models import FastText
            print(f"[FASTTEXT] Loading from: {fasttext_path}")
            fasttext_model = FastText.load(fasttext_path)
            
            self.feature_extractor = FeatureExtractor(fasttext_model)
        
        # Extract features
        features = self.feature_extractor.extract_features([tokens])
        
        print(f"[FEATURES] Shape: {features.shape}")
        
        return features[0]
    
    def predict(
        self,
        text: str,
        model: SentimentModel,
        source: str = 'web',
        ip_address: str = None,
        save_to_db: bool = True
    ) -> Dict:
        """
        Predict sentiment for new text
        
        Args:
            text: Input text
            model: Trained model to use
            source: Source of prediction (web, api, etc)
            ip_address: User IP address
            save_to_db: Whether to save prediction to database
            
        Returns:
            Prediction results
        """
        print("=" * 60)
        print("REAL-TIME PREDICTION")
        print("=" * 60)
        print(f"[INPUT] Text: {text[:100]}...")
        
        import time
        start_time = time.time()
        
        # Load model
        model_data = self.load_model(model)
        pca = model_data['pca']
        svm = model_data['svm']
        
        # Preprocess
        cleaned_text, tokens = self.preprocess_text(text)
        
        # Extract features
        features = self.extract_features(tokens)
        
        # Apply PCA
        print("[PCA] Applying dimensionality reduction...")
        features_pca = pca.transform([features])
        
        # Predict
        print("[SVM] Predicting sentiment...")
        prediction = svm.predict(features_pca)[0]
        probabilities = svm.predict_proba(features_pca)[0]
        
        # Get confidence
        confidence = float(np.max(probabilities))
        
        # Get probabilities for all classes
        class_probabilities = {}
        for i, cls in enumerate(svm.classes_):
            class_probabilities[cls] = float(probabilities[i])
        
        processing_time = time.time() - start_time
        
        print("=" * 60)
        print(f"[RESULT] Sentiment: {prediction}")
        print(f"[RESULT] Confidence: {confidence:.2%}")
        print(f"[RESULT] Time: {processing_time:.3f}s")
        print("=" * 60)
        
        # Save to database
        prediction_record = None
        if save_to_db:
            prediction_record = PredictionRequest.objects.create(
                text=text,
                cleaned_text=cleaned_text,
                model=model,
                predicted_sentiment=prediction,
                confidence_score=confidence,
                source=source,
                ip_address=ip_address
            )
            print(f"[DATABASE] Saved prediction ID: {prediction_record.pk}")
        
        return {
            'text': text,
            'cleaned_text': cleaned_text,
            'tokens': tokens,
            'sentiment': prediction,
            'confidence': confidence,
            'probabilities': class_probabilities,
            'processing_time': processing_time,
            'model_name': model.name,
            'model_accuracy': model.accuracy,
            'prediction_id': prediction_record.pk if prediction_record else None
        }
    
    def get_prediction_history(
        self, 
        model: SentimentModel = None,
        limit: int = 50
    ) -> list:
        """
        Get recent prediction history
        
        Args:
            model: Filter by model (optional)
            limit: Max number of records
            
        Returns:
            List of predictions
        """
        queryset = PredictionRequest.objects.all()
        
        if model:
            queryset = queryset.filter(model=model)
        
        return list(queryset[:limit].values(
            'id',
            'text',
            'predicted_sentiment',
            'confidence_score',
            'source',
            'created_at',
            'user_feedback'
        ))


# Test function
def test_realtime_prediction():
    """Test real-time prediction"""
    print("Testing Real-time Prediction Service...\n")
    
    # This would require a trained model
    # Run manually from Django shell:
    # from core.services.realtime_prediction_service import RealtimePredictionService
    # from core.models import SentimentModel
    # service = RealtimePredictionService()
    # model = SentimentModel.objects.first()
    # result = service.predict("Makanan enak dan pelayanan ramah!", model)
    # print(result)
    
    print("✅ Service initialized. Test from Django shell with trained model.")


if __name__ == '__main__':
    test_realtime_prediction()
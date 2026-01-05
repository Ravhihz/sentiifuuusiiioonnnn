from django.db import models
from django.contrib.auth.models import User
import json


class Dataset(models.Model):
    """Dataset model for storing uploaded data"""

    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    file = models.FileField(upload_to="datasets/")
    uploaded_at = models.DateTimeField(auto_now_add=True)
    uploaded_by = models.ForeignKey(
        User, on_delete=models.CASCADE, null=True, blank=True
    )
    total_reviews = models.IntegerField(default=0)
    is_preprocessed = models.BooleanField(default=False)
    preprocessing_stats = models.JSONField(null=True, blank=True)

    class Meta:
        ordering = ["-uploaded_at"]

    def __str__(self):
        return self.name


class Review(models.Model):
    """Individual review/text"""
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='reviews')
    text = models.TextField()
    
    # Manual label (from labeling interface)
    label = models.CharField(
        max_length=20,
        null=True,
        blank=True,
        choices=[
            ('positive', 'Positive'),
            ('negative', 'Negative'),
            ('neutral', 'Neutral')
        ]
    )
    
    # Predicted label (from batch prediction) - NEW!
    predicted_label = models.CharField(
        max_length=20,
        null=True,
        blank=True,
        choices=[
            ('positive', 'Positive'),
            ('negative', 'Negative'),
            ('neutral', 'Neutral')
        ]
    )
    prediction_confidence = models.FloatField(null=True, blank=True)  # NEW!
    predicted_by_model = models.ForeignKey(
        'SentimentModel',
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name='batch_predictions'
    )  # NEW!
    predicted_at = models.DateTimeField(null=True, blank=True)  # NEW!
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['id']
    
    def __str__(self):
        return f"Review {self.id}: {self.text[:50]}" # type: ignore


class ProcessedReview(models.Model):
    """Preprocessed review with features"""

    review = models.OneToOneField(Review, on_delete=models.CASCADE, related_name="processed")
    cleaned_text = models.TextField()
    tokenized_text = models.TextField()  # JSON array of tokens
    feature_vector = models.TextField()  # JSON array of float values
    created_at = models.DateTimeField(auto_now_add=True)

    def set_tokens(self, tokens):
        """Store tokens as JSON"""
        self.tokenized_text = json.dumps(tokens)

    def get_tokens(self):
        """Get tokens from JSON"""
        return json.loads(self.tokenized_text)

    def set_feature_vector(self, vector):
        """Store feature vector as JSON"""
        self.feature_vector = json.dumps(vector.tolist())

    def get_feature_vector(self):
        """Get feature vector from JSON"""
        import numpy as np
        return np.array(json.loads(self.feature_vector))

    def __str__(self):
        return f"Processed - {self.review}"


class SentimentDictionary(models.Model):
    """Sentiment dictionary storage"""

    name = models.CharField(max_length=255, unique=True)
    description = models.TextField(blank=True)
    dictionary_data = models.JSONField()  # {word: sentiment_score}
    is_extended = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.name


class SentimentModel(models.Model):
    """Trained sentiment analysis model"""
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, related_name='models')
    name = models.CharField(max_length=200)
    algorithm = models.CharField(max_length=50, default='svm')
    
    # Model parameters
    parameters = models.JSONField(default=dict, blank=True)
    
    # Performance metrics
    accuracy = models.FloatField(default=0.0)
    precision = models.FloatField(default=0.0)
    recall = models.FloatField(default=0.0)
    f1_score = models.FloatField(default=0.0)
    
    # Model file path
    model_file = models.CharField(max_length=500)
    
    # Timestamps - MISSING FIELDS!
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    # Additional fields for future use
    trained_by = models.ForeignKey(
        'auth.User', 
        on_delete=models.SET_NULL, 
        null=True, 
        blank=True,
        related_name='trained_models'
    )
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} - {self.accuracy:.2%}"


class TopicModel(models.Model):
    """LDA topic extraction model"""

    sentiment_model = models.ForeignKey(
        SentimentModel, on_delete=models.CASCADE, related_name="topics"
    )
    sentiment_class = models.CharField(max_length=20)
    topic_number = models.IntegerField()
    top_words = models.JSONField()  # [(word, probability), ...]
    document_count = models.IntegerField(default=0)
    perplexity = models.FloatField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["sentiment_class", "topic_number"]

    def __str__(self):
        return f"{self.sentiment_class} - Topic {self.topic_number}"


class TrainingLog(models.Model):
    """Log for training process"""

    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE)
    step = models.CharField(max_length=100)
    message = models.TextField()
    status = models.CharField(
        max_length=20,
        choices=[
            ("info", "Info"),
            ("success", "Success"),
            ("warning", "Warning"),
            ("error", "Error"),
        ],
        default="info",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]

    def __str__(self):
        return f"{self.dataset.name} - {self.step}"
    
class PredictionRequest(models.Model):
    """
    Store real-time prediction requests (production data)
    Separate from training/testing data
    """
    
    # Input
    text = models.TextField()
    cleaned_text = models.TextField(blank=True)
    
    # Model used
    model = models.ForeignKey(
        SentimentModel,
        on_delete=models.CASCADE,
        related_name='predictions'
    )
    
    # Results
    predicted_sentiment = models.CharField(
        max_length=20,
        choices=[
            ('positive', 'Positive'),
            ('negative', 'Negative'),
            ('neutral', 'Neutral')
        ]
    )
    confidence_score = models.FloatField()  # 0.0 - 1.0
    
    # Metadata
    source = models.CharField(max_length=100, default='web')
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    
    # User feedback (optional - for validation)
    user_feedback = models.CharField(
        max_length=20,
        null=True,
        blank=True,
        choices=[
            ('correct', 'Correct'),
            ('incorrect', 'Incorrect')
        ]
    )
    corrected_sentiment = models.CharField(
        max_length=20,
        null=True,
        blank=True,
        choices=[
            ('positive', 'Positive'),
            ('negative', 'Negative'),
            ('neutral', 'Neutral')
        ]
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['-created_at']),
            models.Index(fields=['model', 'predicted_sentiment'])
        ]
    
    def __str__(self):
        return f"{self.predicted_sentiment} ({self.confidence_score:.2f}) - {self.text[:50]}"
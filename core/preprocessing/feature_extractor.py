import numpy as np
from typing import List, Dict, Optional
from gensim.models import FastText


class FeatureExtractor:
    """Extract features from text using FastText word embeddings with sentiment weighting"""
    
    def __init__(self, fasttext_model: FastText, sentiment_dict: Optional[Dict[str, float]] = None, vector_size: int = 300):
        """
        Initialize feature extractor
        
        Args:
            fasttext_model: Trained FastText model
            sentiment_dict: Extended sentiment dictionary {word: sentiment_value}
            vector_size: Dimension of word vectors (default: 300)
        """
        self.model = fasttext_model
        self.vector_size = vector_size
        self.sentiment_dict = sentiment_dict or {}
        
        # Calculate smax (maximum absolute sentiment value)
        if self.sentiment_dict:
            self.smax = max(abs(v) for v in self.sentiment_dict.values())
        else:
            self.smax = 1.0
    
    def calculate_weight(self, word: str) -> float:
        """
        Calculate weight based on sentiment contribution (Formula 3 from paper)
        wi = 2 / (1 + exp(-5 * |s(i)/smax|)) - 1
        
        Args:
            word: Word to calculate weight for
            
        Returns:
            Weight value [0, 1]
        """
        # If no sentiment dictionary or word not in dict, use default weight
        if not self.sentiment_dict or word not in self.sentiment_dict:
            return 1.0
        
        sentiment_value = self.sentiment_dict[word]
        
        # Formula (3): wi = 2 / (1 + exp(-5 * |s(i)/smax|)) - 1
        if self.smax > 0:
            ratio = abs(sentiment_value) / self.smax
            weight = (2.0 / (1.0 + np.exp(-5.0 * ratio))) - 1.0
        else:
            weight = 1.0
        
        return weight
    
    def extract_features(self, tokens: List[str]) -> np.ndarray:
        """
        Extract features from tokens using WEIGHTED averaging (Formula 2 from paper)
        v̄ = Σ(wi × v(di)) / m
        
        Args:
            tokens: List of words/tokens
            
        Returns:
            numpy array of shape (vector_size,)
        """
        if not tokens:
            return np.zeros(self.vector_size)
        
        # Get word vectors and weights for all tokens
        weighted_vectors = []
        
        for token in tokens:
            try:
                # Get word vector from FastText
                vec = self.model.wv[token]
                
                # Calculate weight based on sentiment contribution
                weight = self.calculate_weight(token)
                
                # Weighted vector: wi × v(di)
                weighted_vec = weight * vec
                weighted_vectors.append(weighted_vec)
                
            except KeyError:
                # Skip if word not in vocabulary (shouldn't happen with FastText)
                continue
        
        if not weighted_vectors:
            return np.zeros(self.vector_size)
        
        # Formula (2): v̄ = Σ(wi × v(di)) / m
        feature_vector = np.mean(weighted_vectors, axis=0)
        
        return feature_vector
    
    def extract_batch(self, token_lists: List[List[str]]) -> np.ndarray:
        """
        Extract features for multiple documents
        
        Args:
            token_lists: List of token lists
            
        Returns:
            numpy array of shape (num_docs, vector_size)
        """
        features = []
        for tokens in token_lists:
            feat = self.extract_features(tokens)
            features.append(feat)
        
        return np.array(features)
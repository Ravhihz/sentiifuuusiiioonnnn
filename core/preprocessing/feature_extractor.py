import numpy as np
from typing import List
from gensim.models import FastText


class FeatureExtractor:
    """Extract features from text using FastText word embeddings"""
    
    def __init__(self, fasttext_model: FastText, vector_size: int = 300):
        """
        Initialize feature extractor
        
        Args:
            fasttext_model: Trained FastText model
            vector_size: Dimension of word vectors (default: 300)
        """
        self.model = fasttext_model
        self.vector_size = vector_size
    
    def extract_features(self, tokens: List[str]) -> np.ndarray:
        """
        Extract features from tokens by averaging word vectors
        
        Args:
            tokens: List of words/tokens
            
        Returns:
            numpy array of shape (vector_size,)
        """
        if not tokens:
            return np.zeros(self.vector_size)
        
        # Get word vectors for all tokens
        vectors = []
        for token in tokens:
            try:
                # FastText can handle out-of-vocabulary words
                vec = self.model.wv[token]
                vectors.append(vec)
            except KeyError:
                # Skip if word not in vocabulary (shouldn't happen with FastText)
                continue
        
        if not vectors:
            return np.zeros(self.vector_size)
        
        # Average all word vectors
        feature_vector = np.mean(vectors, axis=0)
        
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
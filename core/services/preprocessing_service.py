import json
from typing import Dict, List
from ..models import Dataset, Review, ProcessedReview
from ..preprocessing import TextPreprocessor
import numpy as np


class PreprocessingService:
    """Service for preprocessing dataset"""

    def __init__(self):
        # Don't initialize feature_extractor here!
        # It needs fasttext_model which we don't have yet
        pass

    def preprocess_dataset(self, dataset: Dataset, use_stemming: bool = True) -> Dict:
        """Preprocess all reviews in dataset with data cleaning"""
        import time
        from collections import defaultdict
        
        start_time = time.time()
        time_breakdown = defaultdict(float)
        
        reviews = dataset.reviews.all() # type: ignore
        total = reviews.count()
        
        if total == 0:
            raise ValueError("No reviews to preprocess")
        
        print(f"[PREPROCESSING] Starting for {total} reviews")
        print(f"[PREPROCESSING] Stemming: {use_stemming}")
        
        # Statistics
        text_lengths_before = []
        text_lengths_after = []
        processed_count = 0
        removed_count = 0
        
        # Track duplicates
        seen_texts = set()
        
        # Step 1: Clean and filter data
        clean_start = time.time()
        valid_reviews = []
        
        duplicate_count = 0
        empty_count = 0
        invalid_count = 0
        
        print(f"[STEP 1] Cleaning data...")
        
        for i, review in enumerate(reviews):
            if (i + 1) % 1000 == 0:
                print(f"  Cleaning: {i + 1}/{total}...")
            
            text = review.text
            
            # Skip empty reviews
            if not text or len(text.strip()) == 0:
                empty_count += 1
                continue
            
            # Check duplicates - faster method
            text_normalized = text.lower().strip()[:200]
            if text_normalized in seen_texts:
                duplicate_count += 1
                continue
            seen_texts.add(text_normalized)
            
            # Skip very short reviews
            if len(text) < 5:
                invalid_count += 1
                continue
            
            valid_reviews.append(review)
            text_lengths_before.append(len(text.split()))
        
        removed_count = empty_count + duplicate_count + invalid_count
        time_breakdown['data_cleaning'] = time.time() - clean_start
        
        print(f"[STEP 1] Complete: {total} → {len(valid_reviews)} reviews")
        print(f"  - Empty: {empty_count}")
        print(f"  - Duplicates: {duplicate_count}")
        print(f"  - Invalid: {invalid_count}")
        
        # Step 2: Text preprocessing
        text_start = time.time()
        print(f"[STEP 2] Preprocessing text...")
        
        preprocessor = TextPreprocessor(use_stemming=use_stemming)
        all_texts = []
        
        # Delete old processed reviews for this dataset
        ProcessedReview.objects.filter(review__dataset=dataset).delete()
        
        for i, review in enumerate(valid_reviews):
            if (i + 1) % 100 == 0:
                print(f"  Processing: {i + 1}/{len(valid_reviews)}...")
            
            # Preprocess text
            cleaned_text = preprocessor.clean_text(review.text)
            cleaned_text = preprocessor.normalize_slang(cleaned_text)
            cleaned_text = preprocessor.remove_stopwords(cleaned_text)
            
            # Tokenize (will apply stemming if use_stemming=True)
            tokens = preprocessor.tokenize(cleaned_text)
            
            # Store for FastText training
            all_texts.append(tokens)
            text_lengths_after.append(len(tokens))
            
            # Save with placeholder feature vector
            ProcessedReview.objects.create(
                review=review,
                cleaned_text=' '.join(tokens),
                tokenized_text=json.dumps(tokens, ensure_ascii=False),
                feature_vector=json.dumps([0.0] * 300)
            )
            
            processed_count += 1
        
        time_breakdown['text_processing'] = time.time() - text_start
        print(f"[STEP 2] Complete: {processed_count} texts preprocessed")
        
        # Step 3: Train FastText
        ft_start = time.time()
        print(f"[STEP 3] Training FastText on {len(all_texts)} documents...")
        
        from gensim.models import FastText
        fasttext_model = FastText(
            sentences=all_texts,
            vector_size=300,
            window=5,
            min_count=1,
            workers=4,
            epochs=5
        )
        
        time_breakdown['fasttext_training'] = time.time() - ft_start
        print(f"[STEP 3] Complete: {time_breakdown['fasttext_training']:.2f}s")
        
        # Step 4: Feature extraction
        fe_start = time.time()
        print(f"[STEP 4] Extracting features...")
        
        # Import and create FeatureExtractor with trained model
        from ..preprocessing.feature_extractor import FeatureExtractor
        feature_extractor = FeatureExtractor(fasttext_model)
        
        # Update with actual feature vectors
        processed_reviews = ProcessedReview.objects.filter(review__dataset=dataset)
        for i, processed in enumerate(processed_reviews):
            if (i + 1) % 100 == 0:
                print(f"  Extracting: {i + 1}/{processed_reviews.count()}...")
            
            tokens = json.loads(processed.tokenized_text)
            features = feature_extractor.extract_features(tokens)
            
            processed.feature_vector = json.dumps(features.tolist())
            processed.save()
        
        time_breakdown['feature_extraction'] = time.time() - fe_start
        print(f"[STEP 4] Complete: {time_breakdown['feature_extraction']:.2f}s")
        
        # Calculate statistics
        avg_length_before = sum(text_lengths_before) / len(text_lengths_before) if text_lengths_before else 0
        avg_length_after = sum(text_lengths_after) / len(text_lengths_after) if text_lengths_after else 0
        
        total_time = time.time() - start_time
        
        # Update dataset
        dataset.is_preprocessed = True
        dataset.preprocessing_stats = { # type: ignore
            'processed': processed_count,
            'removed': removed_count,
            'total_original': total,
            'avg_length_before': avg_length_before,
            'avg_length_after': avg_length_after,
            'use_stemming': use_stemming,
            'processing_time': round(total_time, 2),
            'time_breakdown': {k: round(v, 2) for k, v in time_breakdown.items()}
        }
        dataset.save()
        
        print(f"[PREPROCESSING] Complete! Total: {total_time:.2f}s")
        
        return dataset.preprocessing_stats # type: ignore

    def get_preprocessing_stats(self, dataset: Dataset) -> Dict:
        """Get preprocessing statistics"""
        if not dataset.preprocessing_stats:
            return {}
        return dataset.preprocessing_stats

    def preprocess_single_text(self, text: str, use_stemming: bool = True) -> Dict:
        """Preprocess single text"""
        preprocessor = TextPreprocessor(use_stemming=use_stemming)
        
        cleaned = preprocessor.preprocess(
            text,
            remove_stopwords=True,
            normalize_slang=True,
            stem=use_stemming
        )

        tokens = preprocessor.tokenize(cleaned)

        return {
            'original': text,
            'cleaned': cleaned,
            'tokens': tokens,
            'token_count': len(tokens)
        }
import numpy as np
import json
from typing import Dict, List, Set, Tuple
from collections import Counter
from ..models import Dataset, ProcessedReview


class DictionaryService:
    """Sentiment Dictionary Extension using Cosine Similarity"""
    
    def __init__(self):
        # Initial sentiment dictionary with sentiment values
        # Format: {word: sentiment_value} where sentiment_value in [-1, 1]
        self.sentiment_dict = self._load_initial_dictionary()
    
    def _load_initial_dictionary(self) -> Dict[str, float]:
        """
        Load initial sentiment dictionary
        You can expand this with more words!
        
        Positive words: +0.5 to +1.0
        Negative words: -1.0 to -0.5
        Neutral words: -0.2 to +0.2
        """
        initial_dict = {
            # Positive words
            'bagus': 0.8,
            'baik': 0.7,
            'senang': 0.9,
            'suka': 0.8,
            'mantap': 0.9,
            'hebat': 0.9,
            'luar biasa': 1.0,
            'memuaskan': 0.8,
            'recommended': 0.9,
            'sempurna': 1.0,
            'terbaik': 0.9,
            'cepat': 0.6,
            'ramah': 0.7,
            'nyaman': 0.7,
            'bersih': 0.6,
            
            # Negative words
            'buruk': -0.8,
            'jelek': -0.8,
            'kecewa': -0.9,
            'mengecewakan': -0.9,
            'lambat': -0.6,
            'mahal': -0.5,
            'rusak': -0.8,
            'kotor': -0.7,
            'bau': -0.6,
            'tidak': -0.3,
            'kurang': -0.4,
            'gagal': -0.8,
            'salah': -0.6,
            'complaint': -0.7,
            'komplain': -0.7,
        }
        
        return initial_dict
    
    def bubble_sort_dictionary(self, sentiment_dict: Dict[str, float]) -> List[Tuple[str, float]]:
        """
        Algorithm Line 1: Bubble sort sentiment dictionary in descending order
        
        Returns:
            List of (word, sentiment_value) tuples sorted by sentiment value
        """
        print("[STEP 1] Bubble sorting sentiment dictionary...")
        
        # Convert to list of tuples
        items = list(sentiment_dict.items())
        n = len(items)
        
        # Bubble sort (descending order)
        for i in range(n):
            for j in range(0, n - i - 1):
                if items[j][1] < items[j + 1][1]:  # Compare sentiment values
                    items[j], items[j + 1] = items[j + 1], items[j]
        
        print(f"[SORTED] Top 5 positive: {items[:5]}")
        print(f"[SORTED] Top 5 negative: {items[-5:]}")
        
        return items
    
    def create_reference_set(self, sorted_dict: List[Tuple[str, float]], N: int = 10) -> List[str]:
        """
        Algorithm Line 2: Create reference sentiment words set C
        C = [ND[1:N], ND[length(ND)-N:length(ND)]]
        
        Args:
            sorted_dict: Sorted dictionary from bubble_sort
            N: Number of top/bottom words to select
            
        Returns:
            List of reference words (top N positive + bottom N negative)
        """
        print(f"[STEP 2] Creating reference set with N={N}...")
        
        total_words = len(sorted_dict)
        
        # Top N words (most positive)
        top_n = [word for word, score in sorted_dict[:N]]
        
        # Bottom N words (most negative)
        bottom_n = [word for word, score in sorted_dict[-N:]]
        
        reference_set = top_n + bottom_n
        
        print(f"[REFERENCE SET] {len(reference_set)} words selected")
        print(f"[POSITIVE] {top_n}")
        print(f"[NEGATIVE] {bottom_n}")
        
        return reference_set
    
    def extract_keywords(self, dataset: Dataset) -> Set[str]:
        """
        Algorithm Line 4-10: Extract keywords from review texts
        Keywords = words NOT in current sentiment dictionary
        
        Returns:
            Set of unique keywords
        """
        print("[STEP 3] Extracting keywords from reviews...")
        
        # Get all processed reviews
        processed_reviews = ProcessedReview.objects.filter(review__dataset=dataset)
        
        all_words = []
        for processed in processed_reviews:
            tokens = json.loads(processed.tokenized_text)
            all_words.extend(tokens)
        
        # Count word frequencies
        word_counts = Counter(all_words)
        
        # Extract keywords: words NOT in sentiment dictionary
        keywords = set()
        for word, count in word_counts.items():
            if word not in self.sentiment_dict and count >= 3:  # Minimum 3 occurrences
                keywords.add(word)
        
        print(f"[KEYWORDS] Extracted {len(keywords)} keywords")
        print(f"[SAMPLE] {list(keywords)[:20]}")
        
        return keywords
    
    def get_word_vector(self, word: str, fasttext_model) -> np.ndarray:
        """Get word vector from FastText model"""
        try:
            return fasttext_model.wv[word]
        except KeyError:
            # Return zero vector if word not in vocabulary
            return np.zeros(fasttext_model.vector_size)
    
    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Algorithm Line 14: Calculate cosine similarity
        CS(p)(q) = (v(set(p)) * v(C(q))) / (||v(set(p))|| * ||v(C(q))||)
        
        Args:
            vec1: Vector of keyword
            vec2: Vector of reference word
            
        Returns:
            Cosine similarity value [-1, 1]
        """
        # Avoid division by zero
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        return similarity
    
    def extend_dictionary(
        self, 
        dataset: Dataset, 
        fasttext_model,
        N: int = 10,
        threshold: float = 0.7
    ) -> Dict[str, float]:
        """
        Main algorithm: Extend sentiment dictionary
        
        Args:
            dataset: Dataset object
            fasttext_model: Trained FastText model
            N: Number of reference words (top/bottom N)
            threshold: Similarity threshold (t in algorithm)
            
        Returns:
            Extended dictionary W with new sentiment words
        """
        print("=" * 60)
        print("SENTIMENT DICTIONARY EXTENSION")
        print("=" * 60)
        
        # Step 1: Bubble sort dictionary
        sorted_dict = self.bubble_sort_dictionary(self.sentiment_dict)
        
        # Step 2: Create reference set C
        reference_words = self.create_reference_set(sorted_dict, N=N)
        
        # Step 3: Extract keywords
        keywords = self.extract_keywords(dataset)
        
        if not keywords:
            print("[WARNING] No keywords found!")
            return {}
        
        # Step 4: Calculate similarities and extend dictionary
        print(f"[STEP 4] Calculating cosine similarities (threshold={threshold})...")
        
        extended_dict = {}
        
        for p, keyword in enumerate(keywords, 1):
            if p % 100 == 0:
                print(f"  Processing keyword {p}/{len(keywords)}...")
            
            # Get keyword vector
            keyword_vec = self.get_word_vector(keyword, fasttext_model)
            
            # Calculate similarity with all reference words
            similarities = []
            
            for q, ref_word in enumerate(reference_words, 1):
                ref_vec = self.get_word_vector(ref_word, fasttext_model)
                cs = self.cosine_similarity(keyword_vec, ref_vec)
                similarities.append(cs)
            
            # Get max similarity (Line 16)
            cs_max = max(similarities) if similarities else 0.0
            max_idx = similarities.index(cs_max) if similarities else -1
            
            # If similarity > threshold, add to extended dictionary (Line 17-18)
            if cs_max > threshold:
                # Assign sentiment value based on reference word
                ref_word = reference_words[max_idx]
                sentiment_value = self.sentiment_dict[ref_word]
                
                extended_dict[keyword] = sentiment_value
        
        print(f"[RESULT] Extended dictionary with {len(extended_dict)} new words!")
        print(f"[SAMPLE] {list(extended_dict.items())[:10]}")
        
        return extended_dict
    
    def save_extended_dictionary(self, extended_dict: Dict[str, float], dataset: Dataset):
        """Save extended dictionary to file"""
        import os
        from django.conf import settings
        
        dict_dir = os.path.join(settings.MEDIA_ROOT, 'dictionaries')
        os.makedirs(dict_dir, exist_ok=True)
        
        filename = f'extended_dict_{dataset.pk}.json'
        filepath = os.path.join(dict_dir, filename)
        
        # Merge with original dictionary
        merged_dict = {**self.sentiment_dict, **extended_dict}
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(merged_dict, f, ensure_ascii=False, indent=2)
        
        print(f"[SAVED] Dictionary saved to: {filepath}")
        
        return filepath
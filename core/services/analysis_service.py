import json
from typing import Dict, List, Tuple
from collections import Counter
import numpy as np
from django.db.models import Count
from ..models import Dataset, Review, ProcessedReview


class AnalysisService:
    """Service for preliminary analysis"""

    def __init__(self):
        pass

    def get_dataset_statistics(self, dataset: Dataset) -> Dict:
        """Get comprehensive dataset statistics"""
        reviews = dataset.reviews.all() # type: ignore
        total = reviews.count()

        if total == 0:
            return {'error': 'No reviews in dataset'}

        # Basic stats
        stats = {
            'total_reviews': total,
            'labeled_reviews': reviews.filter(label__isnull=False).count(),
            'unlabeled_reviews': reviews.filter(label__isnull=True).count(),
        }

        # Label distribution
        label_counts = reviews.values('label').annotate(
            count=Count('id')
        )
        stats['label_distribution'] = {
            item['label'] or 'unlabeled': item['count'] 
            for item in label_counts
        }

        # Text length statistics
        text_lengths = [len(r.text.split()) for r in reviews]
        stats['text_length'] = {
            'min': min(text_lengths),
            'max': max(text_lengths),
            'avg': sum(text_lengths) / len(text_lengths),
            'median': sorted(text_lengths)[len(text_lengths) // 2]
        }

        # Word frequency (top 50)
        all_words = []
        for review in reviews:
            words = review.text.lower().split()
            all_words.extend(words)

        word_freq = Counter(all_words).most_common(50)
        stats['top_words'] = [
            {'word': word, 'count': count} 
            for word, count in word_freq
        ]

        return stats

    def get_preprocessed_statistics(self, dataset: Dataset) -> Dict:
        """Get statistics for preprocessed data"""
        processed_reviews = ProcessedReview.objects.filter(
            review__dataset=dataset
        )

        total = processed_reviews.count()

        if total == 0:
            return {'error': 'No preprocessed reviews'}

        # Token statistics
        all_tokens = []
        token_counts = []

        for pr in processed_reviews:
            tokens = json.loads(pr.tokenized_text)
            all_tokens.extend(tokens)
            token_counts.append(len(tokens))

        # Token frequency
        token_freq = Counter(all_tokens).most_common(50)

        stats = {
            'total_processed': total,
            'total_tokens': len(all_tokens),
            'unique_tokens': len(set(all_tokens)),
            'avg_tokens_per_review': sum(token_counts) / len(token_counts),
            'top_tokens': [
                {'token': token, 'count': count}
                for token, count in token_freq
            ]
        }

        return stats

    def get_label_distribution(self, dataset: Dataset) -> Dict:
        """Get detailed label distribution"""
        reviews = dataset.reviews.all() # type: ignore

        distribution = {
            'positive': reviews.filter(label='positive').count(),
            'negative': reviews.filter(label='negative').count(),
            'neutral': reviews.filter(label='neutral').count(),
            'unlabeled': reviews.filter(label__isnull=True).count(),
        }

        total = sum(distribution.values())

        # Calculate percentages
        percentages = {
            label: (count / total * 100) if total > 0 else 0
            for label, count in distribution.items()
        }

        return {
            'counts': distribution,
            'percentages': percentages,
            'total': total
        }

    def get_word_cloud_data(self, dataset: Dataset, label: str = None) -> List[Dict]: # type: ignore
        """Get word frequency data for word cloud"""
        reviews = dataset.reviews.all() # type: ignore

        if label:
            reviews = reviews.filter(label=label)

        # Collect all words
        all_words = []
        for review in reviews:
            words = review.text.lower().split()
            all_words.extend(words)

        # Get frequency
        word_freq = Counter(all_words).most_common(100)

        return [
            {'text': word, 'value': count}
            for word, count in word_freq
        ]

    def check_imbalance(self, dataset: Dataset) -> Dict:
        """Check if dataset is imbalanced"""
        distribution = self.get_label_distribution(dataset)
        counts = distribution['counts']

        # Remove unlabeled
        labeled_counts = {
            k: v for k, v in counts.items() 
            if k != 'unlabeled' and v > 0
        }

        if not labeled_counts:
            return {'is_imbalanced': False, 'message': 'No labeled data'}

        max_count = max(labeled_counts.values())
        min_count = min(labeled_counts.values())

        # Consider imbalanced if ratio > 1.5
        ratio = max_count / min_count if min_count > 0 else float('inf')
        is_imbalanced = ratio > 1.5

        return {
            'is_imbalanced': is_imbalanced,
            'ratio': ratio,
            'max_class': max(labeled_counts, key=labeled_counts.get), # type: ignore
            'min_class': min(labeled_counts, key=labeled_counts.get), # type: ignore
            'max_count': max_count,
            'min_count': min_count,
            'message': f'Dataset is {"imbalanced" if is_imbalanced else "balanced"}. Ratio: {ratio:.2f}'
        }

    def get_raw_data_analysis(self, dataset: Dataset) -> Dict:
        """Analyze raw data before preprocessing"""
        reviews = dataset.reviews.all() # type: ignore
        total = reviews.count()
        
        if total == 0:
            return {'error': 'No reviews in dataset'}
        
        # Detect issues
        empty_reviews = 0
        duplicate_texts = []
        seen_texts = {}
        invalid_chars_count = 0
        
        text_lengths = []
        
        for review in reviews:
            text = review.text
            
            # Empty check
            if not text or text.strip() == '':
                empty_reviews += 1
                continue
            
            # Duplicate check
            text_lower = text.lower().strip()
            if text_lower in seen_texts:
                duplicate_texts.append({
                    'text': text[:100],
                    'count': seen_texts[text_lower] + 1
                })
                seen_texts[text_lower] += 1
            else:
                seen_texts[text_lower] = 1
            
            # Invalid chars check (non-printable)
            if any(ord(char) < 32 or ord(char) > 126 for char in text if ord(char) not in [10, 13]):
                invalid_chars_count += 1
            
            # Length stats
            text_lengths.append(len(text.split()))
        
        # Calculate stats
        avg_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
        
        return {
            'total_reviews': total,
            'valid_reviews': total - empty_reviews,
            'empty_reviews': empty_reviews,
            'duplicate_count': len(duplicate_texts),
            'duplicates': duplicate_texts[:10],  # Top 10
            'invalid_chars_count': invalid_chars_count,
            'avg_text_length': avg_length,
            'min_length': min(text_lengths) if text_lengths else 0,
            'max_length': max(text_lengths) if text_lengths else 0,
        }
    
    def get_data_quality_report(self, dataset: Dataset) -> Dict:
        """Compare before and after preprocessing"""
        if not dataset.is_preprocessed:
            return {'error': 'Dataset not preprocessed yet'}
        
        # Before stats
        raw_stats = self.get_raw_data_analysis(dataset)
        
        # After stats
        processed_reviews = ProcessedReview.objects.filter(review__dataset=dataset)
        
        total_processed = processed_reviews.count()
        removed_count = dataset.total_reviews - total_processed
        
        return {
            'before': raw_stats,
            'after': {
                'total_processed': total_processed,
                'removed_reviews': removed_count,
                'removal_percentage': (removed_count / dataset.total_reviews * 100) if dataset.total_reviews > 0 else 0,
            }
        }
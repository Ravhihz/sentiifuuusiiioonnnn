import numpy as np
from typing import List, Tuple, Dict
from collections import Counter


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculate cosine similarity between two vectors"""
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)

    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0

    return dot_product / (norm_v1 * norm_v2)


def extend_sentiment_dictionary(
    words: List[str],
    word_vectors: Dict[str, np.ndarray],
    sentiment_dict: Dict[str, float],
    top_n: int = 100,
    threshold: float = 0.7,
) -> Dict[str, float]:
    """
    Extend sentiment dictionary using semantic similarity
    Algorithm 1 from the paper
    """
    # Sort sentiment dictionary by sentiment value
    sorted_dict = sorted(sentiment_dict.items(), key=lambda x: abs(x[1]), reverse=True)

    # Get top N positive and negative words
    reference_words = []
    reference_words.extend([w for w, s in sorted_dict[:top_n] if s > 0])
    reference_words.extend([w for w, s in sorted_dict if s < 0][:top_n])

    extended_dict = sentiment_dict.copy()

    for word in words:
        if word in sentiment_dict or word not in word_vectors:
            continue

        max_similarity = 0
        most_similar_word = None

        for ref_word in reference_words:
            if ref_word not in word_vectors:
                continue

            similarity = cosine_similarity(word_vectors[word], word_vectors[ref_word])

            if similarity > max_similarity:
                max_similarity = similarity
                most_similar_word = ref_word

        if max_similarity > threshold and most_similar_word:
            extended_dict[word] = sentiment_dict[most_similar_word] * max_similarity

    return extended_dict


def calculate_sentiment_weight(sentiment_value: float, max_sentiment: float) -> float:
    """
    Calculate sentiment weight based on contribution
    Formula 3 from the paper: w_i = 2 / (1 + e^(-5|s_i/s_max|)) - 1
    """
    if max_sentiment == 0:
        return 1.0

    ratio = abs(sentiment_value / max_sentiment)
    weight = 2 / (1 + np.exp(-5 * ratio)) - 1

    return weight


def weighted_text_vector(
    word_vectors: List[np.ndarray], sentiment_weights: List[float]
) -> np.ndarray:
    """
    Calculate weighted text vector
    Formula 2 from the paper
    """
    if len(word_vectors) == 0:
        return np.zeros(300)

    weighted_sum = np.zeros_like(word_vectors[0])

    for vec, weight in zip(word_vectors, sentiment_weights):
        weighted_sum += weight * vec

    return weighted_sum / len(word_vectors)


def calculate_precision_recall_f1(
    y_true: np.ndarray, y_pred: np.ndarray, labels: List
) -> Dict:
    """Calculate precision, recall, and F1-score for multi-class classification"""
    results = {}

    for label in labels:
        tp = np.sum((y_true == label) & (y_pred == label))
        fp = np.sum((y_true != label) & (y_pred == label))
        fn = np.sum((y_true == label) & (y_pred != label))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": np.sum(y_true == label),
        }

    # Calculate macro average
    macro_precision = np.mean([r["precision"] for r in results.values()])
    macro_recall = np.mean([r["recall"] for r in results.values()])
    macro_f1 = np.mean([r["f1"] for r in results.values()])

    results["macro_avg"] = {
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1,
    }

    return results


def smote_oversampling(
    X: np.ndarray, y: np.ndarray, k: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """SMOTE oversampling for imbalanced dataset"""
    class_counts = Counter(y)
    max_count = max(class_counts.values())

    X_resampled = X.copy()
    y_resampled = y.copy()

    for class_label, count in class_counts.items():
        if count < max_count:
            # Get samples of minority class
            minority_samples = X[y == class_label]
            n_synthetic = max_count - count

            for _ in range(n_synthetic):
                # Random sample
                idx = np.random.randint(0, len(minority_samples))
                sample = minority_samples[idx]

                # Find k nearest neighbors
                distances = np.linalg.norm(minority_samples - sample, axis=1)
                nearest_indices = np.argsort(distances)[1 : k + 1]

                # Random neighbor
                neighbor_idx = np.random.choice(nearest_indices)
                neighbor = minority_samples[neighbor_idx]

                # Generate synthetic sample
                alpha = np.random.random()
                synthetic = sample + alpha * (neighbor - sample)

                X_resampled = np.vstack([X_resampled, synthetic])
                y_resampled = np.append(y_resampled, class_label)

    return X_resampled, y_resampled
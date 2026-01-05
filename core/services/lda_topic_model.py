import numpy as np
from typing import List, Dict, Tuple
from collections import Counter
import json


class LDATopicModel:
    """
    LDA (Latent Dirichlet Allocation) Topic Model
    Discover hidden topics in text data
    No sklearn! Gibbs Sampling implementation!
    """
    
    def __init__(
        self,
        n_topics: int = 5,
        alpha: float = 0.1,
        beta: float = 0.01,
        n_iterations: int = 100,
        random_state: int = 42
    ):
        """
        Initialize LDA model
        
        Args:
            n_topics: Number of topics to extract
            alpha: Document-topic density (lower = fewer topics per doc)
            beta: Topic-word density (lower = fewer words per topic)
            n_iterations: Number of Gibbs sampling iterations
            random_state: Random seed
        """
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.n_iterations = n_iterations
        self.random_state = random_state
        
        # Model parameters (learned during training)
        self.vocab = None  # Vocabulary (unique words)
        self.vocab_size = 0
        self.doc_topic_counts = None  # Document-topic matrix
        self.topic_word_counts = None  # Topic-word matrix
        self.topic_counts = None  # Total words per topic
        self.topics_ = None  # Top words per topic
    
    def fit(self, documents: List[List[str]]) -> 'LDATopicModel':
        """
        Fit LDA model using Gibbs Sampling
        
        Args:
            documents: List of documents (each doc is list of words)
            
        Returns:
            self
        """
        print(f"[LDA] Training LDA with {self.n_topics} topics...")
        print(f"[LDA] Documents: {len(documents)}")
        
        np.random.seed(self.random_state)
        
        # Build vocabulary
        self._build_vocabulary(documents)
        
        # Initialize counts
        n_docs = len(documents)
        self.doc_topic_counts = np.zeros((n_docs, self.n_topics))
        self.topic_word_counts = np.zeros((self.n_topics, self.vocab_size))
        self.topic_counts = np.zeros(self.n_topics)
        
        # Convert documents to word indices
        docs_as_indices = []
        for doc in documents:
            indices = [self.vocab[word] for word in doc if word in self.vocab]
            docs_as_indices.append(indices)
        
        # Random topic assignment
        topic_assignments = []
        for doc_idx, doc in enumerate(docs_as_indices):
            doc_topics = []
            for word_idx in doc:
                # Random topic
                topic = np.random.randint(0, self.n_topics)
                doc_topics.append(topic)
                
                # Update counts
                self.doc_topic_counts[doc_idx, topic] += 1
                self.topic_word_counts[topic, word_idx] += 1
                self.topic_counts[topic] += 1
            
            topic_assignments.append(doc_topics)
        
        # Gibbs Sampling
        print("[LDA] Running Gibbs sampling...")
        
        for iteration in range(self.n_iterations):
            for doc_idx, doc in enumerate(docs_as_indices):
                for word_pos, word_idx in enumerate(doc):
                    # Get current topic
                    topic = topic_assignments[doc_idx][word_pos]
                    
                    # Remove current assignment
                    self.doc_topic_counts[doc_idx, topic] -= 1
                    self.topic_word_counts[topic, word_idx] -= 1
                    self.topic_counts[topic] -= 1
                    
                    # Calculate topic probabilities
                    probs = self._calculate_topic_probabilities(
                        doc_idx, word_idx
                    )
                    
                    # Sample new topic
                    new_topic = np.random.choice(self.n_topics, p=probs)
                    
                    # Update with new topic
                    topic_assignments[doc_idx][word_pos] = new_topic
                    self.doc_topic_counts[doc_idx, new_topic] += 1
                    self.topic_word_counts[new_topic, word_idx] += 1
                    self.topic_counts[new_topic] += 1
            
            if (iteration + 1) % 20 == 0:
                print(f"  Iteration {iteration + 1}/{self.n_iterations}")
        
        print("[LDA] Training complete!")
        
        # Extract topics
        self._extract_topics()
        
        return self
    
    def _build_vocabulary(self, documents: List[List[str]]):
        """Build vocabulary from documents"""
        print("[LDA] Building vocabulary...")
        
        # Collect all words
        all_words = []
        for doc in documents:
            all_words.extend(doc)
        
        # Count word frequencies
        word_counts = Counter(all_words)
        
        # Filter rare words (appear less than 2 times)
        vocab_words = [word for word, count in word_counts.items() if count >= 2]
        
        # Create vocabulary mapping
        self.vocab = {word: idx for idx, word in enumerate(vocab_words)}
        self.vocab_size = len(self.vocab)
        
        print(f"[LDA] Vocabulary size: {self.vocab_size} words")
    
    def _calculate_topic_probabilities(
        self, 
        doc_idx: int, 
        word_idx: int
    ) -> np.ndarray:
        """
        Calculate probability of each topic for given word in document
        
        Formula: P(topic | doc, word) ∝ P(topic | doc) * P(word | topic)
        """
        # P(topic | doc)
        doc_topic_prob = (self.doc_topic_counts[doc_idx] + self.alpha) / \
                        (np.sum(self.doc_topic_counts[doc_idx]) + self.n_topics * self.alpha)
        
        # P(word | topic)
        topic_word_prob = (self.topic_word_counts[:, word_idx] + self.beta) / \
                         (self.topic_counts + self.vocab_size * self.beta)
        
        # Combined probability
        probs = doc_topic_prob * topic_word_prob
        
        # Normalize
        probs /= np.sum(probs)
        
        return probs
    
    def _extract_topics(self, top_n: int = 10):
        """Extract top N words for each topic"""
        print(f"[LDA] Extracting top {top_n} words per topic...")
        
        # Reverse vocabulary (index -> word)
        idx_to_word = {idx: word for word, idx in self.vocab.items()}
        
        self.topics_ = []
        
        for topic_idx in range(self.n_topics):
            # Get word probabilities for this topic
            word_probs = self.topic_word_counts[topic_idx]
            
            # Get top N words
            top_indices = np.argsort(word_probs)[::-1][:top_n]
            
            # Convert to words with probabilities
            top_words = []
            for idx in top_indices:
                word = idx_to_word[idx]
                prob = word_probs[idx] / np.sum(word_probs)
                top_words.append((word, prob))
            
            self.topics_.append(top_words)
            
            # Print topic
            print(f"\n[TOPIC {topic_idx + 1}]")
            for word, prob in top_words:
                print(f"  {word}: {prob:.4f}")
    
    def get_document_topics(self, documents: List[List[str]]) -> np.ndarray:
        """
        Get topic distribution for documents
        
        Args:
            documents: List of documents
            
        Returns:
            Document-topic matrix (n_docs, n_topics)
        """
        n_docs = len(documents)
        doc_topics = np.zeros((n_docs, self.n_topics))
        
        for doc_idx, doc in enumerate(documents):
            for word in doc:
                if word in self.vocab:
                    word_idx = self.vocab[word]
                    
                    # Get topic probabilities for this word
                    topic_probs = self.topic_word_counts[:, word_idx]
                    topic_probs = topic_probs / np.sum(topic_probs)
                    
                    # Add to document topic distribution
                    doc_topics[doc_idx] += topic_probs
            
            # Normalize
            if np.sum(doc_topics[doc_idx]) > 0:
                doc_topics[doc_idx] /= np.sum(doc_topics[doc_idx])
        
        return doc_topics
    
    def get_topics(self) -> List[List[Tuple[str, float]]]:
        """Get discovered topics"""
        return self.topics_
    
    def save_topics(self, filepath: str):
        """Save topics to JSON file"""
        topics_dict = {}
        for topic_idx, words in enumerate(self.topics_):
            topics_dict[f"topic_{topic_idx + 1}"] = [
                {"word": word, "probability": float(prob)}
                for word, prob in words
            ]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(topics_dict, f, ensure_ascii=False, indent=2)
        
        print(f"[LDA] Topics saved to: {filepath}")


# Test function
def test_lda():
    """Test LDA implementation"""
    print("=" * 60)
    print("TESTING LDA TOPIC MODEL")
    print("=" * 60)
    
    # Sample documents
    documents = [
        ['makanan', 'enak', 'lezat', 'restoran', 'bagus'],
        ['makanan', 'lezat', 'harga', 'murah', 'terjangkau'],
        ['pelayanan', 'ramah', 'cepat', 'bagus', 'memuaskan'],
        ['tempat', 'bersih', 'nyaman', 'bagus', 'restoran'],
        ['harga', 'mahal', 'tidak', 'sesuai', 'kualitas'],
        ['pelayanan', 'lambat', 'tidak', 'ramah', 'mengecewakan'],
        ['makanan', 'enak', 'pelayanan', 'bagus', 'recommended'],
        ['tempat', 'kotor', 'tidak', 'nyaman', 'bau'],
    ]
    
    print(f"[TEST] Number of documents: {len(documents)}")
    
    # Train LDA
    lda = LDATopicModel(n_topics=3, n_iterations=50)
    lda.fit(documents)
    
    # Get document topics
    doc_topics = lda.get_document_topics(documents)
    
    print("\n[TEST] Document-Topic Distribution:")
    for doc_idx, topics in enumerate(doc_topics):
        print(f"  Doc {doc_idx + 1}: {topics}")
    
    print("\n✅ LDA test complete!")


if __name__ == '__main__':
    test_lda()
import numpy as np
from typing import List


class LDA:
    """Latent Dirichlet Allocation from scratch using Gibbs Sampling"""

    def __init__(self, n_topics=3, alpha=0.1, beta=0.01, n_iter=1000):
        self.n_topics = n_topics
        self.alpha = alpha
        self.beta = beta
        self.n_iter = n_iter
        self.topic_word_dist = None
        self.doc_topic_dist = None
        self.vocabulary = None

    def fit(self, documents: List[List[int]], vocabulary: List[str]):
        """
        Fit LDA model using Gibbs Sampling

        Args:
            documents: List of documents, each document is list of word indices
            vocabulary: List of unique words
        """
        self.vocabulary = vocabulary
        n_docs = len(documents)
        n_words = len(vocabulary)

        # Initialize count matrices
        doc_topic_count = np.zeros((n_docs, self.n_topics))
        topic_word_count = np.zeros((self.n_topics, n_words))
        topic_count = np.zeros(self.n_topics)

        # Initialize topic assignments randomly
        topic_assignments = []
        for d, doc in enumerate(documents):
            topics = []
            for word_id in doc:
                topic = np.random.randint(0, self.n_topics)
                topics.append(topic)
                doc_topic_count[d, topic] += 1
                topic_word_count[topic, word_id] += 1
                topic_count[topic] += 1
            topic_assignments.append(topics)

        # Gibbs sampling
        for iteration in range(self.n_iter):
            for d, doc in enumerate(documents):
                for w, word_id in enumerate(doc):
                    # Remove current assignment
                    topic = topic_assignments[d][w]
                    doc_topic_count[d, topic] -= 1
                    topic_word_count[topic, word_id] -= 1
                    topic_count[topic] -= 1

                    # Calculate probability for each topic
                    p_topic = np.zeros(self.n_topics)
                    for k in range(self.n_topics):
                        p_topic[k] = (doc_topic_count[d, k] + self.alpha) * (
                            topic_word_count[k, word_id] + self.beta
                        ) / (topic_count[k] + n_words * self.beta)

                    # Normalize
                    p_topic /= np.sum(p_topic)

                    # Sample new topic
                    new_topic = np.random.choice(self.n_topics, p=p_topic)

                    # Update counts
                    topic_assignments[d][w] = new_topic
                    doc_topic_count[d, new_topic] += 1
                    topic_word_count[new_topic, word_id] += 1
                    topic_count[new_topic] += 1

        # Calculate final distributions
        self.doc_topic_dist = (doc_topic_count + self.alpha) / (
            doc_topic_count.sum(axis=1)[:, np.newaxis] + self.n_topics * self.alpha
        )
        self.topic_word_dist = (topic_word_count + self.beta) / (
            topic_word_count.sum(axis=1)[:, np.newaxis] + n_words * self.beta
        )

        return self

    def transform(self, documents):
        """Get topic distribution for documents"""
        return self.doc_topic_dist

    def get_top_words(self, topic_id: int, n_words: int = 10):
        """Get top words for a topic"""
        if self.topic_word_dist is None:
            return []

        word_probs = self.topic_word_dist[topic_id]
        top_word_indices = np.argsort(word_probs)[::-1][:n_words]
        return [(self.vocabulary[i], word_probs[i]) for i in top_word_indices] # type: ignore

    def calculate_perplexity(self, documents: List[List[int]]):
        """Calculate perplexity of the model (Formula 25 from paper)"""
        n_words = sum(len(doc) for doc in documents)
        log_likelihood = 0

        for d, doc in enumerate(documents):
            for word_id in doc:
                word_prob = 0
                for k in range(self.n_topics):
                    word_prob += (
                        self.doc_topic_dist[d, k] * self.topic_word_dist[k, word_id] # type: ignore
                    )
                log_likelihood += np.log(word_prob + 1e-10)

        perplexity = np.exp(-log_likelihood / n_words)
        return perplexity
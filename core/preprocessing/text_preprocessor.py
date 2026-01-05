import re
import json
from pathlib import Path
from typing import List, Optional
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


class TextPreprocessor:
    """Indonesian text preprocessing with optional stemming"""

    def __init__(self, use_stemming: bool = True):
        """
        Initialize text preprocessor
        
        Args:
            use_stemming (bool): Whether to apply stemming (default: True)
        """
        self.use_stemming = use_stemming
        
        # Initialize stemmer (lazy load if needed)
        if self.use_stemming:
            self.stemmer = StemmerFactory().create_stemmer()
        else:
            self.stemmer = None
        
        self.stopwords = self._load_stopwords()
        self.slang_dict = self._load_slang_dict()

    def _load_stopwords(self) -> set:
        """Load Indonesian stopwords"""
        stopwords_path = Path(__file__).parent / "stopwords_id.txt"
        if stopwords_path.exists():
            with open(stopwords_path, "r", encoding="utf-8") as f:
                return set(line.strip() for line in f if line.strip())
        return set()

    def _load_slang_dict(self) -> dict:
        """Load slang dictionary"""
        slang_path = Path(__file__).parent / "slang_dict.json"
        if slang_path.exists():
            with open(slang_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def clean_text(self, text: str) -> str:
        """Basic text cleaning"""
        # Lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)

        # Remove mentions
        text = re.sub(r"@\w+", "", text)

        # Remove hashtags (keep the word)
        text = re.sub(r"#(\w+)", r"\1", text)

        # Keep only alphanumeric and spaces (preserve numbers!)
        text = re.sub(r"[^a-z0-9\s]", " ", text)

        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def normalize_slang(self, text: str) -> str:
        """Normalize Indonesian slang"""
        words = text.split()
        normalized = []

        for word in words:
            if word in self.slang_dict:
                normalized.append(self.slang_dict[word])
            else:
                normalized.append(word)

        return " ".join(normalized)

    def remove_stopwords(self, text: str) -> str:
        """Remove stopwords"""
        words = text.split()
        filtered = [word for word in words if word not in self.stopwords]
        return " ".join(filtered)

    def stem_text(self, text: str) -> str:
        """Stem Indonesian text using Sastrawi (only if enabled)"""
        if self.use_stemming and self.stemmer:
            return self.stemmer.stem(text)
        return text

    def preprocess(
        self,
        text: str,
        remove_stopwords: bool = True,
        normalize_slang: bool = True,
        stem: Optional[bool] = None,
    ) -> str:
        """
        Complete preprocessing pipeline
        
        Args:
            text: Input text
            remove_stopwords: Whether to remove stopwords
            normalize_slang: Whether to normalize slang
            stem: Whether to stem (if None, use instance setting)
        """
        # Clean
        text = self.clean_text(text)

        # Normalize slang
        if normalize_slang:
            text = self.normalize_slang(text)

        # Remove stopwords
        if remove_stopwords:
            text = self.remove_stopwords(text)

        # Stem (respect instance setting if stem param is None)
        if stem is None:
            stem = self.use_stemming
        
        if stem:
            text = self.stem_text(text)

        return text

    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize and optionally stem text
        
        Returns list of tokens (stemmed if use_stemming=True)
        """
        words = text.split()
        
        # Apply stemming only if enabled
        if self.use_stemming and self.stemmer:
            stemmed = [self.stemmer.stem(word) for word in words]
            return stemmed
        else:
            return words

    def preprocess_batch(
        self, texts: List[str], **kwargs
    ) -> List[str]:
        """Preprocess multiple texts"""
        return [self.preprocess(text, **kwargs) for text in texts]
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

class FeatureExtractor:
    def __init__(self, max_features=5000):
        self.vectorizer = TfidfVectorizer(tokenizer=self._tokenizer, max_features=max_features)
        
    def _tokenizer(self, url):
        """Custom tokenizer for URLs."""
        # Split by non-alphanumeric characters
        return re.split(r'[^a-zA-Z0-9]', url)

    def extract_handcrafted_features(self, urls):
        """Extracts features like length, dots, special characters, and HTTPS."""
        features = pd.DataFrame()
        features['url_length'] = urls.apply(len)
        features['num_dots'] = urls.apply(lambda x: x.count('.'))
        features['num_hyphens'] = urls.apply(lambda x: x.count('-'))
        features['num_underscores'] = urls.apply(lambda x: x.count('_'))
        features['num_at'] = urls.apply(lambda x: x.count('@'))
        features['num_question'] = urls.apply(lambda x: x.count('?'))
        features['num_equals'] = urls.apply(lambda x: x.count('='))
        features['num_slash'] = urls.apply(lambda x: x.count('/'))
        features['is_https'] = urls.apply(lambda x: 1 if x.startswith('https') else 0)
        features['num_digits'] = urls.apply(lambda x: sum(c.isdigit() for c in x))
        features['num_alphabets'] = urls.apply(lambda x: sum(c.isalpha() for c in x))
        
        return features

    def fit_transform(self, urls):
        """Fits TF-IDF and extracts all features."""
        tfidf_matrix = self.vectorizer.fit_transform(urls)
        handcrafted = self.extract_handcrafted_features(urls)
        # Combine TF-IDF with handcrafted features
        # Note: Handcrafted features are converted to sparse matrix to stack with tfidf_matrix
        return hstack([tfidf_matrix, handcrafted.values])

    def transform(self, urls):
        """Transforms URLs using fitted TF-IDF and extracts features."""
        tfidf_matrix = self.vectorizer.transform(urls)
        handcrafted = self.extract_handcrafted_features(urls)
        return hstack([tfidf_matrix, handcrafted.values])

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Optional Deep Learning Imports for Cloud Compatibility
try:
    import tensorflow as tf
    import tf_keras as keras
    from tf_keras.models import Sequential
    from tf_keras.layers import Embedding, LSTM, Dense, Dropout, SpatialDropout1D
    from tf_keras.preprocessing.text import Tokenizer
    from tf_keras.preprocessing.sequence import pad_sequences
    from transformers import BertTokenizer, TFBertForSequenceClassification
    DL_AVAILABLE = True
except ImportError:
    DL_AVAILABLE = False

def get_ml_models():
    """Returns a dictionary of ML models."""
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, n_jobs=-1),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1)
    }

def build_lstm_model(input_dim, num_classes, max_len=64):
    """Builds an LSTM model using Keras."""
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=128, input_length=max_len),
        SpatialDropout1D(0.2),
        LSTM(100, dropout=0.2, recurrent_dropout=0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def build_bert_model(num_classes, model_name='bert-base-uncased'):
    """Builds a BERT-based model using HuggingFace."""
    model = TFBertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
    optimizer = keras.optimizers.Adam(learning_rate=2e-5)
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    return model

class LSTMProcessor:
    def __init__(self, max_len=64, num_words=10000):
        self.tokenizer = Tokenizer(num_words=num_words, char_level=True)
        self.max_len = max_len

    def fit_transform(self, texts):
        self.tokenizer.fit_on_texts(texts)
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, maxlen=self.max_len)

    def transform(self, texts):
        sequences = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(sequences, maxlen=self.max_len)

class BERTProcessor:
    def __init__(self, model_name='bert-base-uncased', max_len=64):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_len = max_len

    def encode(self, texts):
        return self.tokenizer(
            texts.tolist(),
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='tf'
        )

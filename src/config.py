import os

# Paths
DATA_DIR = "data"
RESULTS_DIR = "results"
PLOTS_DIR = "plots"

# Dataset filename (Expected ISCX-URL2016 or similar CSV)
DATASET_FILE = os.path.join(DATA_DIR, "malicious_phish.csv")

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Labels
CLASSES = ['benign', 'phishing', 'malware', 'defacement']

# Features configuration
TFIDF_MAX_FEATURES = 5000
BERT_MODEL_NAME = "bert-base-uncased"
MAX_LEN = 64 # Max length for LSTM and BERT tokens
BATCH_SIZE = 32
EPOCHS = 5

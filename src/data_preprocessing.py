import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import re

def load_data(file_path):
    """Loads the dataset and performs basic cleaning."""
    if not os.path.exists(file_path):
        print(f"Warning: Dataset not found at {file_path}. Creating synthetic data for demonstration.")
        return create_synthetic_data()
    
    df = pd.read_csv(file_path)
    # Ensure columns are named correctly (typical for sid321axn dataset)
    if 'url' in df.columns and 'type' in df.columns:
        df = df.rename(columns={'type': 'label'})
    
    # Cleaning: Remove duplicates and nulls
    df = df.dropna()
    df = df.drop_duplicates()
    
    # Filter for requested classes if necessary
    target_classes = ['benign', 'phishing', 'malware', 'defacement']
    df = df[df['label'].isin(target_classes)]
    
    return df

def clean_url(url):
    """Cleans the URL string."""
    url = url.lower()
    url = re.sub(r'https?://', '', url)
    url = re.sub(r'www\.', '', url)
    return url

def preprocess_data(df):
    """Clean URLs and encode labels."""
    df['cleaned_url'] = df['url'].apply(clean_url)
    
    le = LabelEncoder()
    df['label_encoded'] = le.fit_transform(df['label'])
    
    return df, le

def create_synthetic_data():
    """Creates a small synthetic dataset if the real one is missing."""
    data = {
        'url': [
            'http://www.google.com', 'https://facebook.com', 'http://malware-site.net/virus.exe',
            'http://phishing-bank.com/login', 'http://defaced-gov.org/index.html',
            'http://safe-shopping.com', 'http://attacker-server.com/payload',
            'http://login-verify-account.com', 'http://hacked-blog.com/update',
            'http://legit-site.co.uk', 'http://malicious-link.ru/run',
            'http://bank-secure-login.com/auth', 'http://government-portal.gov.dz/home',
            'http://free-gift-cards.top/claim', 'http://system-update-fix.exe',
            'http://university-portal.edu/student', 'http://secure-payment-verify.com',
            'http://hacker-zone.cn/exploit', 'http://news-official.com/today',
            'http://phish-login.com', 'http://malware-drop.com', 'http://deface-test.com'
        ],
        'label': [
            'benign', 'benign', 'malware', 'phishing', 'defacement',
            'benign', 'malware', 'phishing', 'defacement',
            'benign', 'malware', 'phishing', 'benign',
            'phishing', 'malware', 'benign', 'phishing',
            'malware', 'benign', 'phishing', 'malware', 'defacement'
        ]
    }
    return pd.DataFrame(data)

def split_data(df, test_size=0.2, random_state=42):
    """Splits data into train and test sets."""
    return train_test_split(
        df['url'], df['label_encoded'], 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df['label_encoded']
    )

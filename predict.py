import sys
import os
import joblib
import pandas as pd
from src.data_preprocessing import clean_url

def predict_url(url):
    # Paths
    model_path = os.path.join("models", "best_rf_model.pkl")
    extractor_path = os.path.join("models", "feature_extractor.pkl")
    le_path = os.path.join("models", "label_encoder.pkl")
    
    if not os.path.exists(model_path):
        print("Error: Model not found. Please run 'python main.py' first to train and save the models.")
        return

    # Load resources
    model = joblib.load(model_path)
    extractor = joblib.load(extractor_path)
    le = joblib.load(le_path)

    # Preprocess the input URL
    cleaned_url = clean_url(url)
    
    # Extract features
    # (Note: transform expects a Series/list)
    features = extractor.transform(pd.Series([url]))
    
    # Predict
    prediction_idx = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    
    # Get labels
    label = le.inverse_transform([prediction_idx])[0]
    confidence = probabilities[prediction_idx] * 100
    
    print(f"\nURL: {url}")
    print(f"Classification: {label.upper()}")
    print(f"Confidence: {confidence:.2f}%")
    
    print("\nAll Probabilities:")
    for i, class_name in enumerate(le.classes_):
        print(f" - {class_name}: {probabilities[i]*100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <url>")
        sys.exit(1)
    
    input_url = sys.argv[1]
    predict_url(input_url)

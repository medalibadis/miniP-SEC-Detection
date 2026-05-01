import pandas as pd
import numpy as np
import os
import argparse
import tensorflow as tf
from src.config import *
from src.data_preprocessing import load_data, preprocess_data, split_data
from src.feature_engineering import FeatureExtractor
from src.models import get_ml_models, build_lstm_model, build_bert_model, LSTMProcessor, BERTProcessor
from src.evaluation import evaluate_model, plot_confusion_matrix, plot_roc_curve, plot_model_comparison
from src.optimization import optimize_xgboost

def main(run_bert=False, optimize_xgb=False):
    # 1. Data Loading and Preprocessing
    print("Step 1: Loading and Preprocessing Data...")
    df = load_data(DATASET_FILE)
    df, le = preprocess_data(df)
    X_train_raw, X_test_raw, y_train, y_test = split_data(df, TEST_SIZE, RANDOM_STATE)
    
    classes = list(le.classes_)
    num_classes = len(classes)
    
    # 2. Feature Extraction
    print("Step 2: Extracting Features...")
    extractor = FeatureExtractor(max_features=TFIDF_MAX_FEATURES)
    X_train_ml = extractor.fit_transform(X_train_raw)
    X_test_ml = extractor.transform(X_test_raw)
    
    results = []

    # 3. ML Models Training and Evaluation
    print("Step 3: Training ML Models...")
    ml_models = get_ml_models()
    
    for name, model in ml_models.items():
        print(f"Training {name}...")
        
        # Optimization for XGBoost if requested
        if name == 'XGBoost' and optimize_xgb:
            model = optimize_xgboost(X_train_ml, y_train)
            name = "XGBoost (Optimized)"
            
        model.fit(X_train_ml, y_train)
        y_pred = model.predict(X_test_ml)
        y_prob = model.predict_proba(X_test_ml)
        
        metrics = evaluate_model(y_test, y_pred, y_prob, le, name)
        results.append(metrics)
        
        plot_confusion_matrix(y_test, y_pred, classes, name, PLOTS_DIR)
        plot_roc_curve(y_test, y_prob, classes, name, PLOTS_DIR)

    # 4. DL Models: LSTM
    print("Step 4: Training LSTM Model...")
    lstm_proc = LSTMProcessor(max_len=MAX_LEN)
    X_train_lstm = lstm_proc.fit_transform(X_train_raw)
    X_test_lstm = lstm_proc.transform(X_test_raw)
    
    lstm_model = build_lstm_model(input_dim=len(lstm_proc.tokenizer.word_index) + 1, 
                                 num_classes=num_classes, 
                                 max_len=MAX_LEN)
    
    lstm_model.fit(X_train_lstm, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)
    
    y_prob_lstm = lstm_model.predict(X_test_lstm)
    y_pred_lstm = np.argmax(y_prob_lstm, axis=1)
    
    metrics_lstm = evaluate_model(y_test, y_pred_lstm, y_prob_lstm, le, "LSTM")
    results.append(metrics_lstm)
    plot_confusion_matrix(y_test, y_pred_lstm, classes, "LSTM", PLOTS_DIR)
    plot_roc_curve(y_test, y_prob_lstm, classes, "LSTM", PLOTS_DIR)

    # 5. Bonus: BERT-based model
    if run_bert:
        print("Step 5 (Bonus): Training BERT Model...")
        bert_proc = BERTProcessor(model_name=BERT_MODEL_NAME, max_len=MAX_LEN)
        X_train_bert = bert_proc.encode(X_train_raw)
        X_test_bert = bert_proc.encode(X_test_raw)
        
        bert_model = build_bert_model(num_classes=num_classes, model_name=BERT_MODEL_NAME)
        
        # Train BERT (Note: This is very slow on CPU)
        bert_model.fit(
            {'input_ids': X_train_bert['input_ids'], 'attention_mask': X_train_bert['attention_mask']},
            y_train,
            epochs=2, # Keep it small for demonstration
            batch_size=8 # Small batch size for BERT
        )
        
        # Predict
        logits = bert_model.predict(
            {'input_ids': X_test_bert['input_ids'], 'attention_mask': X_test_bert['attention_mask']}
        ).logits
        y_prob_bert = tf.nn.softmax(logits).numpy()
        y_pred_bert = np.argmax(y_prob_bert, axis=1)
        
        metrics_bert = evaluate_model(y_test, y_pred_bert, y_prob_bert, le, "BERT")
        results.append(metrics_bert)
        plot_confusion_matrix(y_test, y_pred_bert, classes, "BERT", PLOTS_DIR)
        plot_roc_curve(y_test, y_prob_bert, classes, "BERT", PLOTS_DIR)

    # 6. Comparison and Results
    print("\nStep 6: Comparison and Final Results...")
    results_df = pd.DataFrame(results)
    print("\nModel Comparison Table:")
    print(results_df.sort_values(by='accuracy', ascending=False))
    
    results_df.to_csv(os.path.join(RESULTS_DIR, "model_results.csv"), index=False)
    plot_model_comparison(results_df, PLOTS_DIR)

    # Feature Importance for Random Forest
    rf_model = ml_models.get('Random Forest')
    if rf_model:
        print("Plotting Feature Importance...")
        tfidf_features = list(extractor.vectorizer.get_feature_names_out())
        handcrafted_features = ['url_length', 'num_dots', 'num_hyphens', 'num_underscores', 'num_at', 
                                'num_question', 'num_equals', 'num_slash', 'is_https', 'num_digits', 'num_alphabets']
        all_feature_names = tfidf_features + handcrafted_features
        from src.evaluation import plot_feature_importance
        plot_feature_importance(rf_model, all_feature_names, PLOTS_DIR)

    # 7. Save the Best Model (Random Forest)
    print("\nStep 7: Saving the Best Model (Random Forest)...")
    import joblib
    model_path = os.path.join("models", "best_rf_model.pkl")
    extractor_path = os.path.join("models", "feature_extractor.pkl")
    le_path = os.path.join("models", "label_encoder.pkl")
    
    # Save the RF model if it was trained
    rf_model = ml_models.get('Random Forest')
    if rf_model:
        joblib.dump(rf_model, model_path)
        joblib.dump(extractor, extractor_path)
        joblib.dump(le, le_path)
        print(f"Model saved to {model_path}")

    print(f"\nAll plots saved to {PLOTS_DIR}/")
    print(f"Results table saved to {RESULTS_DIR}/model_results.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Malicious URL Detection using AI')
    parser.add_argument('--bert', action='store_true', help='Run BERT model (Bonus)')
    parser.add_argument('--optimize', action='store_true', help='Perform GridSearchCV on XGBoost')
    args = parser.parse_args()
    
    main(run_bert=args.bert, optimize_xgb=args.optimize)

# Malicious URL Detection using AI

This project implements a multi-class classification system to detect malicious URLs using various Machine Learning and Deep Learning techniques.

## Features
- **Data Loading & Preprocessing**: Cleans URLs and encodes labels (Benign, Phishing, Malware, Defacement).
- **Feature Extraction**:
    - **TF-IDF Vectorization**: Capture character-level/word-level patterns.
    - **Handcrafted Features**: URL length, number of dots, special characters (@, -, _), use of HTTPS, etc.
- **Models**:
    - **Machine Learning**: Logistic Regression, Random Forest, XGBoost.
    - **Deep Learning**: LSTM (using Keras/TensorFlow).
    - **Bonus**: BERT-based model (using HuggingFace Transformers).
- **Optimization**: GridSearchCV for XGBoost hyperparameter tuning.
- **Evaluation**: Accuracy, Precision, Recall, F1-score, Confusion Matrix, and ROC-AUC curves.

## Project Structure
```
.
├── main.py                 # Main entry point
├── requirements.txt        # Dependencies
├── src/                    # Source code modules
│   ├── config.py           # Configuration and constants
│   ├── data_preprocessing.py # Data loading and cleaning
│   ├── feature_engineering.py # TF-IDF and handcrafted features
│   ├── models.py           # Model definitions
│   ├── optimization.py     # GridSearchCV logic
│   └── evaluation.py       # Metrics and plotting logic
├── results/                # Saved results (CSV)
├── plots/                  # Saved plots (PNG)
├── manuel_realisation.pdf  # Final Report (PDF)
└── README.md               # This file
```

## Setup and Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Dataset Note**:
   The `malicious_phish.csv` dataset is not included in this submission due to its size. To run the code, please place the dataset in a `data/` folder in the root directory. 
│   └── evaluation.py       # Metrics and plotting logic
├── results/                # Saved results (CSV)
└── plots/                  # Saved plots (PNG)
```

## Setup and Installation

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Dataset**:
   Download the **ISCX-URL2016** (or the [Malicious URLs Dataset from Kaggle](https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset)) and rename it to `malicious_phish.csv`. Place it in the `data/` directory.
   
   *Note: If the file is missing, the code will run with a small synthetic dataset for demonstration.*

## Running the Project

Run the full pipeline with default settings:
```bash
python main.py
```

Run with XGBoost Optimization:
```bash
python main.py --optimize
```

Run with BERT Model (Warning: Slow on CPU):
```bash
python main.py --bert
```

## Deliverables
- **Code Source**: All Python files in this repository.
- **Compte Rendu**: Refer to the generated results in the `results/` and `plots/` directories for your manual.

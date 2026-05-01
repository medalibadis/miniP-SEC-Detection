from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import pandas as pd

def optimize_xgboost(X_train, y_train):
    """Performs GridSearchCV on XGBoost."""
    print("\nStarting GridSearchCV for XGBoost (this may take a while)...")
    
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5],
        'learning_rate': [0.1, 0.2],
        'gamma': [0, 0.1]
    }
    
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_jobs=-1)
    
    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=3,
        scoring='accuracy',
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

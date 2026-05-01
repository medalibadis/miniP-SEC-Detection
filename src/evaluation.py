import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import numpy as np
import os

def evaluate_model(y_true, y_pred, y_prob, label_encoder, model_name):
    """Calculates metrics and prints them."""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    print(f"\n--- {model_name} Evaluation ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    metrics = {
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    return metrics

def plot_confusion_matrix(y_true, y_pred, classes, model_name, save_path):
    """Plots and saves the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'cm_{model_name.lower().replace(" ", "_")}.png'))
    plt.close()

def plot_roc_curve(y_true, y_prob, classes, model_name, save_path):
    """Plots and saves the ROC curve for multi-class."""
    # Binarize labels for multi-class ROC
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    plt.figure(figsize=(10, 8))
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
        plt.plot(fpr, tpr, label=f'{classes[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'roc_{model_name.lower().replace(" ", "_")}.png'))
    plt.close()

def plot_model_comparison(results_df, save_path):
    """Plots comparison of all models."""
    plt.figure(figsize=(12, 6))
    df_melted = results_df.melt(id_vars='model', var_name='Metric', value_name='Score')
    sns.barplot(data=df_melted, x='model', y='Score', hue='Metric')
    plt.title('Model Performance Comparison')
    plt.ylim(0, 1.1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'model_comparison.png'))
    plt.close()

def plot_feature_importance(model, feature_names, save_path, top_n=20):
    """Plots and saves the most important features."""
    if not hasattr(model, 'feature_importances_'):
        return
        
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]
    
    plt.figure(figsize=(10, 8))
    plt.title(f'Top {top_n} Most Important Features')
    plt.barh(range(len(indices)), importances[indices], color='b', align='center')
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'feature_importance.png'))
    plt.close()

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_curve, auc)
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_image_data(data_dir, target_size=(64,64)):
    """
    Carica immagini da directory strutturata per classi.
    Restituisce X_flat (N, features), y (N,), class_names
    """
    transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    X_tensor, y = next(iter(loader))
    X = X_tensor.numpy()  # (N, C, H, W)
    N, C, H, W = X.shape
    X_flat = X.reshape(N, C * H * W)
    return X_flat, np.array(y), dataset.classes


# Percorsi e parametri
DATA_DIR = './augmented_dataset'  # Directory con immagini organizzate per classi
OUTPUT_DIR = 'results'
os.makedirs(OUTPUT_DIR, exist_ok=True)
METRICS_JSON = os.path.join(OUTPUT_DIR, 'metrics.json')
LC_PLOT = os.path.join(OUTPUT_DIR, 'learning_curves.png')
ROC_PLOT = os.path.join(OUTPUT_DIR, 'roc_curve.png')

# 1) Caricamento dati
X, y, class_names = load_image_data(DATA_DIR, target_size=(64,64))

# 2) Split
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42)

# 3) Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# 4) Modelli
def get_models():
    return {
        'SVC': SVC(kernel='rbf', probability=True, random_state=42),
        'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=42),
        'RF': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGB': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }
models = get_models()

# 5) Learning Curves
def plot_and_save_learning_curves(models, X, y, output_path):
    plt.figure(figsize=(12,8))
    for name, model in models.items():
        train_sizes, train_scores, val_scores = learning_curve(
            model, X, y, cv=5, train_sizes=np.linspace(0.1,1.0,5),
            scoring='accuracy', n_jobs=-1)
        train_mean = np.mean(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        plt.plot(train_sizes, train_mean, label=f'{name} Train')
        plt.plot(train_sizes, val_mean, '--', label=f'{name} Val')
    plt.title('Learning Curves')
    plt.xlabel('Training examples')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

plot_and_save_learning_curves(models, X_train_scaled, y_train, LC_PLOT)

# 6) Training finale e metriche
data_results = {}

# one-hot per multiclass ROC
lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)

for name, model in models.items():
    # fit su train+val
    X_tv = np.vstack([X_train_scaled, X_val_scaled])
    y_tv = np.hstack([y_train, y_val])
    model.fit(X_tv, y_tv)

    # previsioni
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    # metriche base
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted')
    }

    # ROC e AUC
    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
        fpr, tpr, _ = roc_curve(y_test, y_proba[:,1])
        roc_auc = auc(fpr, tpr)
        metrics.update({'fpr': fpr.tolist(), 'tpr': tpr.tolist(), 'auc': roc_auc})
    else:
        # multiclass macro AUC
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_proba.ravel())
        roc_auc = auc(fpr, tpr)
        metrics.update({'auc_macro': roc_auc})

    data_results[name] = metrics

# 7) Plot ROC multi-modello
def plot_and_save_roc(data_results, output_path):
    plt.figure(figsize=(8,6))
    for name, m in data_results.items():
        if 'fpr' in m:
            plt.plot(m['fpr'], m['tpr'], label=f"{name} (AUC {m['auc']:.3f})")
    plt.plot([0,1],[0,1],'k--')
    plt.title('ROC Curves')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

plot_and_save_roc(data_results, ROC_PLOT)

# 8) Salvataggio metriche in JSON
with open(METRICS_JSON, 'w') as f:
    json.dump(data_results, f, indent=4)

print(f"Saved learning curves to {LC_PLOT}")
print(f"Saved ROC curves to {ROC_PLOT}")
print(f"Saved metrics to {METRICS_JSON}")

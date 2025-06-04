import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_curve, auc, classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib

# Configuration
DATA_DIR = "./augmented_dataset"  # Root directory containing subfolders 'Mosquito' and 'Not_Mosquito'
IMAGE_SIZE = (128, 128)  # Resize dimensions
TEST_SIZE = 0.2
RANDOM_STATE = 42
METRICS_OUTPUT = "metrics_results.csv"
MODELS_OUTPUT_DIR = "models"

# 1. Load and preprocess images
def load_images(data_dir, image_size):
    X = []
    y = []
    labels = os.listdir(data_dir)
    for label in labels:
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in os.listdir(label_dir):
            fpath = os.path.join(label_dir, fname)
            try:
                img = Image.open(fpath).convert('RGB')
                img = img.resize(image_size)
                arr = np.array(img).flatten()
                X.append(arr)
                y.append(label)
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
    return np.array(X), np.array(y)

# Load data
print("\n --- Loading images from dataset --- ")
X, y = load_images(DATA_DIR, IMAGE_SIZE)

# Encode labels
print("\n --- Encoding labels --- ")
encoder = LabelEncoder()
y_enc = encoder.fit_transform(y)

# Split train/test
print("\n --- Splitting dataset into train and test sets --- ")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_enc
)

# Standardize features
print("\n --- Standardizing features --- ")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Ensure output dirs
os.makedirs(MODELS_OUTPUT_DIR, exist_ok=True)

# 2. Define models
def get_models():
    models = {
        'SVC': SVC(probability=True, random_state=RANDOM_STATE, class_weight='balanced', kernel='linear', max_iter=1000 , verbose=True),
        'MLP': MLPClassifier(random_state=RANDOM_STATE, early_stopping=True, max_iter=500, learning_rate='adaptive', verbose=True),
        'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE, verbose=True),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=RANDOM_STATE, device='cuda', verbosity=1, early_stopping_rounds=10)
    }
    return models


models = get_models()
""" 
# 3. Train, evaluate, and collect metrics
print("\n --- Training and evaluating models --- ")
metrics_list = []
plt.figure(figsize=(10, 8))

for name, model in models.items():
    ''' 
    print(f"Training {name}...")
    if name == 'XGBoost':
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=True)
    else:
        model.fit(X_train, y_train)
    # Save model
    joblib.dump(model, os.path.join(MODELS_OUTPUT_DIR, f"{name}.joblib"))
    '''
    # load pre-trained model if exists
    model_path = os.path.join(MODELS_OUTPUT_DIR, f"{name}.joblib")
    if os.path.exists(model_path):
        print(f"Loading pre-trained model for {name}...")
        model = joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Model file {model_path} does not exist. Please train the model first.")

    print(f"\n --- Evaluating {name} ---")
    # Predictions and probabilities
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Classification metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # ROC and AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

    # Log metrics
    metrics_list.append({
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'AUC': roc_auc
    })

# Plot ROC
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

# 4. Save metrics to CSV
metrics_df = pd.DataFrame(metrics_list)
metrics_df.to_csv(METRICS_OUTPUT, index=False)
print(f"Metrics saved to {METRICS_OUTPUT}")
 """
# 5. Detailed classification report saved
def save_reports(models, X_test, y_test, encoder, output_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)
    for name, model in models.items():
        # load pre-trained model if exists
        model_path = os.path.join(MODELS_OUTPUT_DIR, f"{name}.joblib")
        if os.path.exists(model_path):
            print(f"Loading pre-trained model for {name}...")
            model = joblib.load(model_path)
        else:
            raise FileNotFoundError(f"Model file {model_path} does not exist. Please train the model first.")

        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, target_names=encoder.classes_)
        with open(os.path.join(output_dir, f"report_{name}.txt"), 'w') as f:
            f.write(report)

save_reports(models, X_test, y_test, encoder)
print("Classification reports saved in 'reports/' directory")

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import time
import os
import copy
import numpy as np
from PIL import Image
import torch
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.utils.multiclass import unique_labels
import json
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class CustomTensorDataset(Dataset):
    """Dataset personalizzato per dati e etichette in memoria (NumPy/Tensor)."""
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = torch.from_numpy(labels).long() if isinstance(labels, np.ndarray) else torch.tensor(labels).long()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        if isinstance(sample, np.ndarray):
            # Cast a uint8
            if sample.dtype in [np.float32, np.float64]:
                sample_uint8 = (sample * 255.0).clip(0, 255).astype(np.uint8)
            else:
                sample_uint8 = sample.astype(np.uint8)

            # PIL conversion
            if sample_uint8.ndim == 2:
                pil_img = Image.fromarray(sample_uint8, mode='L')
            elif sample_uint8.ndim == 3 and sample_uint8.shape[2] == 1:
                pil_img = Image.fromarray(sample_uint8.squeeze(-1), mode='L')
            elif sample_uint8.ndim == 3 and sample_uint8.shape[2] == 3:
                pil_img = Image.fromarray(sample_uint8)
            else:
                raise ValueError(f"Formato immagine non supportato: shape={sample_uint8.shape}")
        else:
            pil_img = sample

        if self.transform:
            pil_img = self.transform(pil_img)

        return pil_img, label

class TransferLearning:
    def __init__(self, X_train, y_train, X_val, y_val,
                 num_classes:int=2, batch_size=32, num_epochs=15,
                 learning_rate=0.001, models_names=None,
                 need_resize:bool=True, need_normalize:bool=True,
                 early_stop_patience:int=10, models_dir:str='saved_models'):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.models_names = models_names or ['ResNet18','ResNet50','MobileNetV2']
        self.need_resize = need_resize
        self.need_normalize = need_normalize
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.early_stop_patience = early_stop_patience
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        print(f"Utilizzo del dispositivo: {self.device}")

    def prepare_data(self):
        self.data_transforms = {
            'train': transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor()]),
            'val': transforms.Compose([transforms.ToTensor()])
        }
        if self.need_resize:
            self.data_transforms['train'] = transforms.Compose([
                transforms.Resize((224,224)), *self.data_transforms['train'].transforms
            ])
            self.data_transforms['val'] = transforms.Compose([
                transforms.Resize((224,224)), *self.data_transforms['val'].transforms
            ])
        if self.need_normalize:
            self.data_transforms['train'] = transforms.Compose([
                *self.data_transforms['train'].transforms,
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])
            self.data_transforms['val'] = transforms.Compose([
                *self.data_transforms['val'].transforms,
                transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
            ])

        train_dataset = CustomTensorDataset(self.X_train, self.y_train, transform=self.data_transforms['train'])
        val_dataset   = CustomTensorDataset(self.X_val,   self.y_val,   transform=self.data_transforms['val'])

        self.dataloaders = {
            'train': DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,  num_workers=0),
            'val':   DataLoader(val_dataset,   batch_size=self.batch_size, shuffle=False, num_workers=0)
        }
        self.dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}
        print(f"Dataset sizes: {self.dataset_sizes}")

    def get_model(self, model_name_str, num_classes_val, pretrained=True):
        model_ft = None
        if model_name_str == 'ResNet18':
            model_ft = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        elif model_name_str == 'ResNet50':
            model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        elif model_name_str == 'MobileNetV2':
            model_ft = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError("Model name non valido")

        for param in model_ft.parameters(): param.requires_grad = False
        if model_name_str.startswith('ResNet'):
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes_val)
        else:
            num_ftrs = model_ft.classifier[1].in_features
            model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes_val)
        return model_ft.to(self.device)

    def train_model_instance(self, model, criterion, optimizer):
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        epochs_no_improve = 0
        history = {'train_loss':[], 'val_loss':[], 'train_acc':[], 'val_acc':[]}
        since = time.time()

        for epoch in range(self.num_epochs):
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            for phase in ['train','val']:
                model.train() if phase=='train' else model.eval()
                running_loss=0.0; running_corrects=0
                for inputs, labels in self.dataloaders[phase]:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase=='train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs,1)
                        loss = criterion(outputs, labels)
                        if phase=='train': loss.backward(); optimizer.step()
                    running_loss += loss.item()*inputs.size(0)
                    running_corrects += torch.sum(preds==labels.data)
                epoch_loss = running_loss/self.dataset_sizes[phase]
                epoch_acc = running_corrects.double()/self.dataset_sizes[phase]
                history[f"{phase}_loss"].append(epoch_loss)
                history[f"{phase}_acc"].append(epoch_acc.item())
                print(f" {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

                if phase=='val':
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                        epochs_no_improve = 0
                    else:
                        epochs_no_improve += 1

            if epochs_no_improve >= self.early_stop_patience:
                print(f"Early stopping after {epoch+1} epochs without improvement.")
                break
            print()

        time_elapsed = time.time() - since
        print(f"Training completed in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
        model.load_state_dict(best_model_wts)
        return model, history, best_acc.item()

    def save_model(self, model, model_name, accuracy):
        filename = f"{model_name}_{accuracy:.4f}.pth"
        path = os.path.join(self.models_dir, filename)
        torch.save(model.state_dict(), path)
        print(f"Model saved to {path}")

    def run_transfer_learning(self):
        self.prepare_data()
        self.all_histories = {}
        self.final_accuracies = {}

        for model_name in self.models_names:
            print(f"\nTraining {model_name}")
            model = self.get_model(model_name, self.num_classes)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=self.learning_rate)
            trained_model, history, best_acc = self.train_model_instance(model, criterion, optimizer)
            self.all_histories[model_name] = history
            self.final_accuracies[model_name] = best_acc
            self.save_model(trained_model, model_name, best_acc)

    def show_training_results(self):
        print("\n--- Risultati Finali ---")
        for name, acc in self.final_accuracies.items():
            print(f"{name}: {acc:.4f}")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,5))
        for name, h in self.all_histories.items():
            epochs_i = range(1, len(h['train_loss']) + 1)
            ax1.plot(epochs_i, h['train_loss'], label=f"{name} train")
            ax1.plot(epochs_i, h['val_loss'], '--',    label=f"{name} val")
            ax2.plot(epochs_i, h['train_acc'], label=f"{name} train")
            ax2.plot(epochs_i, h['val_acc'], '--',    label=f"{name} val")
        ax1.set(title='Loss', xlabel='Epoca', ylabel='Loss')
        ax1.legend(); ax1.grid()
        ax2.set(title='Accuratezza', xlabel='Epoca', ylabel='Acc')
        ax2.legend(); ax2.grid()
        plt.tight_layout(); plt.show()



'''
--- Risultati Finali ---
ResNet18: 0.8659
ResNet50: 0.8665
MobileNetV2: 0.8674
'''
def evaluate_and_save_results(X_test, y_test, models_dir='saved_models', num_classes=2,
                              batch_size=32, need_resize=True, need_normalize=True, device=None,
                              output_json='test_results.json', roc_plot_path='roc_curves.png', label_map=None):
    """
    Carica tutti i modelli salvati in models_dir, valuta su X_test e y_test,
    calcola le metriche di classificazione (accuracy, precision, recall, f1), ROC e AUC,
    e salva le prestazioni in JSON e le curve ROC in un file PNG.
    """

    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

    transforms_list = []
    if need_resize:
        transforms_list.append(transforms.Resize((224, 224)))
    transforms_list.append(transforms.ToTensor())
    if need_normalize:
        transforms_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    test_transform = transforms.Compose(transforms_list)

    test_dataset = CustomTensorDataset(X_test, y_test, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    results = {}
    all_fpr, all_tpr = {}, {}
    y_true_all = []
    y_prob_all = {}

    for filename in os.listdir(models_dir):
        if filename.endswith('.pth'):
            model_name = filename[:-4]
            if '_' in model_name:
                name, acc_str = model_name.rsplit('_', 1)
            else:
                name, acc_str = model_name, ''

            try:
                model = TransferLearning(None, None, None, None, num_classes=num_classes).get_model(name, num_classes)
                path = os.path.join(models_dir, filename)
                model.load_state_dict(torch.load(path, map_location=device))
                model.to(device)
                model.eval()

                y_true = []
                y_pred = []
                y_probs = []

                with torch.no_grad():
                    for inputs, labels in test_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        outputs = model(inputs)
                        probs = torch.softmax(outputs, dim=1)
                        _, predicted = torch.max(probs, 1)

                        y_true.extend(labels.cpu().numpy())
                        y_pred.extend(predicted.cpu().numpy())
                        y_probs.extend(probs.cpu().numpy())

                y_true_np = np.array(y_true)
                y_pred_np = np.array(y_pred)
                y_probs_np = np.array(y_probs)

                acc = accuracy_score(y_true_np, y_pred_np)
                prec = precision_score(y_true_np, y_pred_np, average='weighted')
                rec = recall_score(y_true_np, y_pred_np, average='weighted')
                f1 = f1_score(y_true_np, y_pred_np, average='weighted')
                class_labels = unique_labels(y_true_np, y_pred_np)
                f1_per_class = f1_score(y_true_np, y_pred_np, average=None)

                # Usa i nomi reali, es: Mosquito / Not_Mosquito (devi passarli, vedi nota sotto)
                f1_per_class_dict = {
                    label_map[int(class_idx)] if label_map else str(class_idx): float(score)
                    for class_idx, score in zip(class_labels, f1_per_class)
                }

                #model_results['f1_per_class'] = f1_per_class_dict


                model_results = {
                    'filename': filename,
                    'loaded_accuracy': float(acc_str) if acc_str else None,
                    'test_accuracy': acc,
                    'precision': prec,
                    'recall': rec,
                    'f1_score': f1,
                    'f1_per_class': f1_per_class_dict
                }

                # ROC and AUC
                y_true_bin = label_binarize(y_true_np, classes=np.arange(num_classes))
                if num_classes == 2:
                    fpr, tpr, _ = roc_curve(y_true_np, y_probs_np[:, 1])
                    model_results['roc_auc'] = auc(fpr, tpr)
                    all_fpr[name] = fpr
                    all_tpr[name] = tpr
                else:
                    fpr = dict()
                    tpr = dict()
                    roc_auc = dict()
                    for i in range(num_classes):
                        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_probs_np[:, i])
                        roc_auc[i] = auc(fpr[i], tpr[i])
                    model_results['roc_auc'] = roc_auc

                results[name] = model_results

            except Exception as e:
                print(f"Errore caricando {filename}: {e}")

    with open(output_json, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Risultati di test salvati in {output_json}")

    # Salva curva ROC (solo per binario)
    if num_classes == 2:
        plt.figure(figsize=(8,6))
        for name in all_fpr:
            plt.plot(all_fpr[name], all_tpr[name], label=f"{name} (AUC = {results[name]['roc_auc']:.2f})")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves')
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(roc_plot_path)
        plt.close()
        print(f"Curve ROC salvate in {roc_plot_path}")

    return results


def evaluate_sklearn_model(model_original, X_train, y_train, X_val, y_val):
    """
    Riaddestra un modello scikit-learn con gli stessi iperparametri,
    valuta su (X_val, y_val) e restituisce:
      - il modello riaddestrato
      - metrics: dict con accuracy, precision, recall, f1_weighted, f1_per_class
    """
    from copy import deepcopy
    
    model_class = type(model_original)
    params = model_original.get_params()
    print(f"  [sklearn] Riaddestro {model_class.__name__} con params: {params}")
    
    # Crea nuova istanza e riaddestra su X_train
    model = model_class(**params)
    if model_class.__name__ == "XGBClassifier":
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)
    else:
        # Per altri modelli, non serve eval_set     
        model.fit(X_train, y_train)
    
    # Predizioni su validation
    y_pred = model.predict(X_val)
    
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_val, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_val, y_pred, average="weighted", zero_division=0)
    
    f1_per_class_arr = f1_score(y_val, y_pred, average=None, zero_division=0)
    f1_per_class = {
        f"class_{i}": float(score)
        for i, score in enumerate(f1_per_class_arr)
    }
    
    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_weighted": float(f1_w),
        "f1_per_class": f1_per_class,
    }
    return model, metrics

def evaluate_transfer_model(
    model, train_dataset, val_dataset, device, epochs=3, batch_size=32
):
    """
    Riaddestra solo l'ultimo strato di un modello PyTorch (transfer learning),
    utilizzando train_dataset (Dataset PyTorch) e val_dataset.
    Restituisce:
      - il modello riaddestrato
      - metrics: dict con accuracy, precision, recall, f1_weighted, f1_per_class
    """
    # Congela tutti i parametri
    for param in model.parameters():
        param.requires_grad = False
    
    # Individua l'ultimo classificatore (fc o classifier)
    if hasattr(model, "fc"):
        for p in model.fc.parameters():
            p.requires_grad = True
        classifier = model.fc
    elif hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True
        classifier = model.classifier
    else:
        raise ValueError("Modello non compatibile: nessun attributo 'fc' o 'classifier' trovato")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Ciclo di training solo sul classificatore finale
    model.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
    
    # Valutazione
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            out = model(xb)
            preds = out.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(yb.cpu().numpy())
    
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    f1_per_class_arr = f1_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = {
        f"class_{i}": float(score) for i, score in enumerate(f1_per_class_arr)
    }
    
    metrics = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_weighted": float(f1_w),
        "f1_per_class": f1_per_class,
    }
    return model, metrics


def generate_learning_curves(
    X_flat_train, y_flat_train,
    X_flat_val, y_flat_val,
    X_train_full, y_train_full,
    X_val, y_val,
    sklearn_models_dir,
    pytorch_models_dir,
    get_transfer_model,
    output_plot_path='learning_curves.png',
    output_metrics_path='learning_curve_metrics.json',
    output_models_dir='retrained_models'
):
    """
    Riaddestra progressivamente modelli scikit-learn e PyTorch (transfer learning),
    calcola per ogni frazione di training set le metriche (accuracy, precision, recall, f1 weighted, f1 per classe),
    salva i modelli riaddestrati completi, salva il plot delle accuracy, e scrive tutte le metriche in JSON.

    Parametri:
        X_train_full  : array-like (N_train × features), train set completo
        y_train_full  : array-like (N_train,), etichette del train set
        X_val         : array-like (N_val × features), validation set fisso
        y_val         : array-like (N_val,), etichette del validation set
        sklearn_models_dir  : path alla cartella con modelli .joblib da riaddestrare
        pytorch_models_dir  : path alla cartella con modelli .pth (transfer learning)
        get_transfer_model  : funzione lambda(name, num_classes) → modello PyTorch con classificatore reinizializzato
        output_plot_path    : path per salvare il grafico delle sole accuracy curve
        output_metrics_path : path per salvare il JSON con tutte le metriche
        output_models_dir   : cartella dove salvare i modelli riaddestrati (100% train)
    """
    os.makedirs(output_models_dir, exist_ok=True)

    X_train_full = np.array(X_train_full)
    y_train_full = np.array(y_train_full)
    X_val = np.array(X_val)
    y_val = np.array(y_val)

    # Percentuali di training set da usare (ad es. 10%, 32%, 55%, 77%, 100%)
    train_sizes = np.linspace(0.1, 1.0, 5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dizionario per raccogliere tutte le metriche
    metrics_all = {}

    # 1) Modelli scikit-learn
    for fname in os.listdir(sklearn_models_dir):
        if not fname.endswith(".joblib"):
            continue
        model_path = os.path.join(sklearn_models_dir, fname)
        model_original = joblib.load(model_path)
        model_name = fname.replace(".joblib", "")
        print(f"\n[SKLEARN] Processing: {model_name}")

        metrics_all[model_name] = {}
        acc_curve = []

        for frac in train_sizes:
            pct_label = f"{int(frac*100)}%"
            n = int(len(X_flat_train) * frac)
            X_sub = X_flat_train[:n]
            y_sub = y_flat_train[:n]

            model, metrics = evaluate_sklearn_model(
                model_original, X_sub, y_sub, X_flat_val, y_flat_val
            )
            metrics_all[model_name][pct_label] = metrics
            acc_curve.append(metrics["accuracy"])

            if frac == 1.0:
                out_path = os.path.join(output_models_dir, f"{model_name}_retrained.joblib")
                joblib.dump(model, out_path)

        metrics_all[model_name]["_accuracy_curve"] = acc_curve

    # 2) Modelli PyTorch (transfer learning)
    for fname in os.listdir(pytorch_models_dir):
        if not fname.endswith(".pth"):
            continue
        name = fname.replace(".pth", "")
        print(f"\n[PyTorch] Processing: {name}")

        metrics_all[name] = {}
        accuracy_curve = []

        for frac in train_sizes:
            pct_label = f"{int(frac * 100)}%"
            n_samples = int(len(X_train_full) * frac)
            X_sub = X_train_full[:n_samples]
            y_sub = y_train_full[:n_samples]

            X_tensor = torch.tensor(X_sub, dtype=torch.float32)
            y_tensor = torch.tensor(y_sub, dtype=torch.long)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long)

            model = get_transfer_model(name, num_classes=2)
            model, metrics = evaluate_transfer_model(
                model, X_tensor, y_tensor, X_val_tensor, y_val_tensor, device
            )
            metrics_all[name][pct_label] = metrics
            accuracy_curve.append(metrics["accuracy"])

            # Salva il modello completo solo quando frac == 1.0
            if frac == 1.0:
                out_path = os.path.join(output_models_dir, f"{name}_retrained.pth")
                torch.save(model.state_dict(), out_path)

        metrics_all[name]["_accuracy_curve"] = accuracy_curve

    # 3) Salvataggio del grafico delle only-accuracy curves
    plt.figure(figsize=(10, 7))
    for name, data in metrics_all.items():
        curve = data["_accuracy_curve"]
        plt.plot(train_sizes * 100, curve, label=name)
    plt.xlabel('Training set size (%)')
    plt.ylabel('Validation Accuracy')
    plt.title('Learning Curves (Accuracy)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_plot_path)
    plt.close()
    print(f"\n✅ Grafico salvato in: {output_plot_path}")

    # Rimuoviamo le curve _accuracy_curve prima di salvare il JSON
    for name in metrics_all:
        del metrics_all[name]["_accuracy_curve"]

    # 4) Salvataggio di tutte le metriche in JSON
    with open(output_metrics_path, 'w') as f:
        json.dump(metrics_all, f, indent=4)
    print(f"✅ Metriche complete salvate in: {output_metrics_path}")

    return metrics_all


def load_dataset(data_dir, image_size=(128, 128)):
    """
    Scorre le sottocartelle di `data_dir` (una per ciascuna label),
    estrae:
      - X_flat: array NumPy (N × (H*W*3)) con immagini resize & flatten
      - file_paths: lista di path assoluti alle immagini (per PyTorch)
      - y: array di etichette numeriche
      - encoder: LabelEncoder per convertire nome_label ↔ indice
    """
    flat_list = []
    paths_list = []
    labels_list = []
    
    for label in sorted(os.listdir(data_dir)):
        label_dir = os.path.join(data_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in os.listdir(label_dir):
            fpath = os.path.join(label_dir, fname)
            try:
                img = Image.open(fpath).convert("RGB")
                img = img.resize(image_size)
                arr = np.array(img).flatten()
                flat_list.append(arr)
                paths_list.append(fpath)
                labels_list.append(label)
            except Exception as e:
                print(f"Warning: impossibile caricare {fpath}: {e}")
    
    X_flat = np.vstack(flat_list)  # (N, H*W*3)
    y_labels = np.array(labels_list)  # (N,)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_labels)  # (N,) numerico
    return X_flat, paths_list, y, encoder

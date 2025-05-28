import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms # Rimosso datasets perché non usiamo ImageFolder
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import time
import os
import copy
import numpy as np
from PIL import Image # Necessario per convertire array NumPy in immagini PIL se le transform lo richiedono

class CustomTensorDataset(Dataset):
    """Dataset personalizzato per dati e etichette in memoria (NumPy/Tensor)."""
    def __init__(self, data, labels, transform=None):
        # data: array-like di shape (N, H, W, C) o (N, H, W)
        # labels: array-like di lunghezza N
        self.data = data
        self.labels = torch.from_numpy(labels).long() if isinstance(labels, np.ndarray) else torch.tensor(labels).long()
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]

        # Se il dato è un array NumPy, converti a uint8 e poi a PIL Image
        if isinstance(sample, np.ndarray):
            # Se float, assumiamo in [0,1], riconverti a [0,255]
            if sample.dtype in [np.float32, np.float64]:
                sample_uint8 = (sample * 255.0).clip(0, 255).astype(np.uint8)
            else:
                sample_uint8 = sample.astype(np.uint8)

            # Se scala di grigi HxW o HxWx1
            if sample_uint8.ndim == 2:
                pil_img = Image.fromarray(sample_uint8, mode='L')
            elif sample_uint8.ndim == 3 and sample_uint8.shape[2] == 1:
                pil_img = Image.fromarray(sample_uint8.squeeze(-1), mode='L')
            elif sample_uint8.ndim == 3 and sample_uint8.shape[2] == 3:
                pil_img = Image.fromarray(sample_uint8)
            else:
                raise ValueError(f"Formato immagine non supportato: shape={sample_uint8.shape}")
        else:
            # se già PIL Image o altro
            pil_img = sample

        # Applica eventuali transformazioni di torchvision
        if self.transform:
            pil_img = self.transform(pil_img)

        return pil_img, label
    
    
class TransferLearning:
    def __init__(self, X_train, y_train, X_val, y_val, num_classes:int=2, batch_size=32, num_epochs=15, learning_rate=0.001,
                models_names=['ResNet18', 'ResNet50', 'MobileNetV2'], need_resize:bool=True, need_normalize:bool=True):
        """
        Inizializza la classe TransferLearning con i dati di addestramento e validazione.
        
        Args:
            X_train (array-like): Dati di addestramento.
            y_train (array-like): Etichette di addestramento.
            X_val (array-like): Dati di validazione.
            y_val (array-like): Etichette di validazione.
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.models_names = models_names
        self.need_resize = need_resize  # Aggiunto per gestire il resize condizionale
        self.need_normalize = need_normalize
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Utilizzo del dispositivo: {self.device}")


    def prepare_data(self):
        self.data_transforms = {
            'train': transforms.Compose([
                # Resize è incluso qui per sicurezza. Rimuovi se X_train ha già immagini 224x224.
                #transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),  # Data augmentation
                transforms.ToTensor(),
                # Normalize è incluso qui. Rimuovi se X_train è GIÀ normalizzato con stats ImageNet.
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                # Resize è incluso qui per sicurezza. Rimuovi se X_val ha già immagini 224x224.
                #transforms.Resize((224, 224)),
                transforms.ToTensor(),
                # Normalize è incluso qui. Rimuovi se X_val è GIÀ normalizzato con stats ImageNet.
                #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
        }
        if self.need_resize:
            # Aggiungi Resize se necessario
            self.data_transforms['train'] = transforms.Compose([
                transforms.Resize((224, 224)),
                *self.data_transforms['train'].transforms
            ])
            self.data_transforms['val'] = transforms.Compose([
                transforms.Resize((224, 224)),
                *self.data_transforms['val'].transforms
            ])
        if self.need_normalize:
            # Aggiungi Normalize se necessario
            self.data_transforms['train'] = transforms.Compose([
                *self.data_transforms['train'].transforms,
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            self.data_transforms['val'] = transforms.Compose([
                *self.data_transforms['val'].transforms,
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        # Crea le Dataset personalizzate
        try:
            train_dataset = CustomTensorDataset(self.X_train, self.y_train, transform=self.data_transforms['train'])
            val_dataset = CustomTensorDataset(self.X_val, self.y_val, transform=self.data_transforms['val'])
        except NameError:
            print("ERRORE: Le variabili X_train, y_train, X_val, y_val non sono definite.")
            print("Per favore, sostituisci i placeholder all'inizio dello script con i tuoi dati.")
            exit()


        # Creazione dei DataLoader
        self.dataloaders = {
            'train': DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0), # num_workers>0 può dare problemi su Windows o in certi notebook
            'val': DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        }
        self.dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset)}

        print(f"Dimensioni del dataset di Training: {self.dataset_sizes['train']}")
        print(f"Dimensioni del dataset di Validazione: {self.dataset_sizes['val']}")

    # --- 3. Definizione dei Modelli (identica alla versione precedente) ---
    def get_model(self, model_name_str, num_classes_val, pretrained=True):
        model_ft = None
        if model_name_str == 'ResNet18':
            model_ft = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        elif model_name_str == 'ResNet50':
            model_ft = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        elif model_name_str == 'MobileNetV2':
            model_ft = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            print("Nome modello non valido")
            return None

        for param in model_ft.parameters():
            param.requires_grad = False

        if model_name_str.startswith('ResNet'):
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes_val)
        elif model_name_str == 'MobileNetV2':
            num_ftrs = model_ft.classifier[1].in_features
            model_ft.classifier[1] = nn.Linear(num_ftrs, num_classes_val)
        
        model_ft = model_ft.to(self.device)
        return model_ft

    # --- 4. Funzione di Addestramento (identica alla versione precedente) ---
    def train_model_instance(self, model, criterion, optimizer, num_epochs=None):
        num_epochs = num_epochs if num_epochs is not None else self.num_epochs
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

        for epoch in range(num_epochs):
            print(f'Epoca {epoch+1}/{num_epochs}')
            print('-' * 10)
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                running_loss = 0.0
                running_corrects = 0
                for inputs, labels in self.dataloaders[phase]: # Qui inputs sono già tensori trasformati
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = running_corrects.double() / self.dataset_sizes[phase]
                print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
                if phase == 'train':
                    history['train_loss'].append(epoch_loss)
                    history['train_acc'].append(epoch_acc.item())
                else:
                    history['val_loss'].append(epoch_loss)
                    history['val_acc'].append(epoch_acc.item())
                    if epoch_acc > best_acc:
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
            print()
        time_elapsed = time.time() - since
        print(f'Addestramento completato in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Migliore Acc Valutazione: {best_acc:.4f}')
        model.load_state_dict(best_model_wts)
        return model, history, best_acc.item()

    # --- 5. Esecuzione dell'Addestramento per Ciascun Modello (identica) ---
    def run_transfer_learning(self):
        self.prepare_data()  # Prepara i dati e i DataLoader
        print("\n--- Inizio Addestramento dei Modelli ---")
        self.all_histories = {}
        self.final_accuracies = {}

        for model_name_str in self.models_names:
            print(f"\n--- Addestramento del modello: {model_name_str} ---")
            model_ft = self.get_model(model_name_str, self.num_classes)
            if model_ft is None: continue
            criterion = nn.CrossEntropyLoss()
            params_to_update = filter(lambda p: p.requires_grad, model_ft.parameters())
            optimizer_ft = optim.Adam(params_to_update, lr=self.learning_rate)
            trained_model, history, best_val_acc = self.train_model_instance(model_ft, criterion, optimizer_ft, num_epochs=self.num_epochs)
            self.all_histories[model_name_str] = history
            self.final_accuracies[model_name_str] = best_val_acc

    # --- 6. Valutazione e Grafici di Confronto (identica) ---
    def show_training_results(self):    
        print("\n--- Risultati Finali ---")
        for model_name_val, acc in self.final_accuracies.items():
            print(f"Accuratezza finale di validazione per {model_name_val}: {acc:.4f}")

        epochs_range = range(1, self.num_epochs + 1)
        plt.figure(figsize=(15, 6))
        plt.subplot(1, 2, 1)
        for model_name_plot, history_plot in self.all_histories.items():
            plt.plot(epochs_range, history_plot['train_loss'], label=f'{model_name_plot} Train Loss')
            plt.plot(epochs_range, history_plot['val_loss'], linestyle='--', label=f'{model_name_plot} Val Loss')
        plt.title('Confronto Loss di Addestramento e Validazione')
        plt.xlabel('Epoche')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.subplot(1, 2, 2)
        for model_name_plot, history_plot in self.all_histories.items():
            plt.plot(epochs_range, history_plot['train_acc'], label=f'{model_name_plot} Train Accuracy')
            plt.plot(epochs_range, history_plot['val_acc'], linestyle='--', label=f'{model_name_plot} Val Accuracy')
        plt.title('Confronto Accuratezza di Addestramento e Validazione')
        plt.xlabel('Epoche')
        plt.ylabel('Accuratezza')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

import tensorflow as tf
from tensorflow.keras import layers, models, applications, optimizers
import matplotlib.pyplot as plt
import seaborn as sns # Per grafici più belli (matrice di confusione)
import time
import os
import numpy as np
import json
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Assicuriamoci che la GPU venga usata correttamente (best practice)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU trovata e Memory Growth abilitato per: {gpus}")
    except RuntimeError as e:
        print(f"Errore nell'impostare la Memory Growth: {e}")

class TransferLearning:
    def __init__(self, train_dir, val_dir, test_dir,
                 num_classes:int=2, batch_size=32, num_epochs=15,
                 learning_rate=0.001, models_names=None,
                 early_stop_patience:int=10, models_dir:str='keras_models',
                 results_dir:str='keras_models_performances'):
        
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.models_names = models_names or ['ResNet50', 'MobileNetV2', 'NASNetMobile']
        self.early_stop_patience = early_stop_patience
        self.models_dir = models_dir
        self.results_dir = results_dir # Nuova cartella per i risultati
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.img_height = 224
        self.img_width = 224
        
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None
        self.class_names = []
        self.all_histories = {}
        self.final_accuracies = {}

    def prepare_data(self):
        # ... questo metodo è identico alla versione precedente ...
        print("--- Fase 1: Caricamento dati dal disco ---")
        
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            self.train_dir, label_mode='int', seed=123,
            image_size=(self.img_height, self.img_width), batch_size=self.batch_size)

        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            self.val_dir, label_mode='int', seed=123,
            image_size=(self.img_height, self.img_width), batch_size=self.batch_size)

        self.test_ds = tf.keras.utils.image_dataset_from_directory(
            self.test_dir, label_mode='int', seed=123,
            image_size=(self.img_height, self.img_width), batch_size=self.batch_size, shuffle=False)
            
        self.class_names = self.train_ds.class_names
        print(f"Classi trovate: {self.class_names}")

        AUTOTUNE = tf.data.AUTOTUNE
        self.train_ds = self.train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)
        self.test_ds = self.test_ds.cache().prefetch(buffer_size=AUTOTUNE)


    def get_model(self, model_name_str):
        # ... questo metodo è identico alla versione precedente ...
        print(f"--- Fasi 2 & 3: Creazione modello {model_name_str} con preprocessing integrato ---")
        
        input_shape = (self.img_height, self.img_width, 3)
        inputs = tf.keras.Input(shape=input_shape)

        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"), layers.RandomRotation(0.1), layers.RandomZoom(0.1),
        ], name='data_augmentation')
        x = data_augmentation(inputs)

        if model_name_str == 'ResNet50':
            preprocess_input = applications.resnet50.preprocess_input
            base_model = applications.ResNet50(input_shape=input_shape, include_top=False, weights='imagenet')
        elif model_name_str == 'MobileNetV2':
            preprocess_input = applications.mobilenet_v2.preprocess_input
            base_model = applications.MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet')
        elif model_name_str == 'NASNetMobile':
            preprocess_input = applications.nasnet.preprocess_input
            base_model = applications.NASNetMobile(input_shape=input_shape, include_top=False, weights='imagenet')
        else:
            raise ValueError(f"Nome del modello non supportato: {model_name_str}")
        
        x = preprocess_input(x)
        base_model.trainable = False
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = models.Model(inputs, outputs)
        return model

    def evaluate_model(self, model, model_name):
        """
        NUOVO METODO: Esegue una valutazione completa del modello sul test set
        e salva metriche e grafici.
        """
        print(f"\n--- Inizio Valutazione Completa di {model_name} sul Test Set ---")

        # 1. Ottenere le predizioni e le etichette reali
        y_probs = model.predict(self.test_ds) # Probabilità
        y_pred = np.argmax(y_probs, axis=1)   # Classi predette
        y_true = np.concatenate([y for x, y in self.test_ds], axis=0) # Etichette vere

        # Dizionario per salvare tutti i risultati
        results = {}

        # 2. Calcolare e salvare il Classification Report
        print("\nClassification Report:")
        report_dict = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        report_str = classification_report(y_true, y_pred, target_names=self.class_names)
        print(report_str)
        results['classification_report'] = report_dict

        # 3. Creare e salvare la Matrice di Confusione
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title(f'Matrice di Confusione - {model_name}')
        plt.ylabel('Etichetta Reale')
        plt.xlabel('Etichetta Predetta')
        cm_path = os.path.join(self.results_dir, f'confusion_matrix_{model_name}.png')
        plt.savefig(cm_path)
        plt.close()
        print(f"\nMatrice di confusione salvata in: {cm_path}")
        results['confusion_matrix_path'] = cm_path

        # 4. Creare e salvare la Curva ROC (gestisce caso binario e multi-classe)
        if self.num_classes == 2:
            fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
            roc_auc = auc(fpr, tpr)
            results['roc_auc'] = roc_auc
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'Curva ROC - {model_name}')
            plt.legend(loc="lower right")
            roc_path = os.path.join(self.results_dir, f'roc_curve_{model_name}.png')
            plt.savefig(roc_path)
            plt.close()
            print(f"Curva ROC salvata in: {roc_path}")
            results['roc_curve_path'] = roc_path
        else:
            # Per il multi-classe, l'AUC è spesso calcolato come media
            print("\nLa curva ROC per il multi-classe non è stata implementata in questo esempio.")

        # 5. Salvare tutte le metriche in un file JSON
        json_path = os.path.join(self.results_dir, f'evaluation_metrics_{model_name}.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Metriche di valutazione complete salvate in: {json_path}")


    def run_transfer_learning(self):
        """Esegue il ciclo completo di addestramento e valutazione."""
        self.prepare_data()

        for model_name in self.models_names:
            print(f"\n===== INIZIO ADDESTRAMENTO: {model_name} =====")
            
            model = self.get_model(model_name)
            
            model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate),
                          loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=self.early_stop_patience, restore_best_weights=True)
            ]

            since = time.time()
            history = model.fit(
                self.train_ds, epochs=self.num_epochs, validation_data=self.val_ds, callbacks=callbacks)
            time_elapsed = time.time() - since
            print(f"Addestramento completato in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
            
            self.all_histories[model_name] = history.history
            best_acc = max(history.history['val_accuracy'])
            self.final_accuracies[model_name] = best_acc
            
            filename = f"{model_name}_{best_acc:.4f}.keras"
            path = os.path.join(self.models_dir, filename)
            model.save(path)
            print(f"Modello salvato in: {path}")

            # --- CHIAMATA AL NUOVO METODO DI VALUTAZIONE ---
            if self.test_ds:
                self.evaluate_model(model, model_name)

    def save_training_results(self, show_plots=False):
        # ... questo metodo è identico alla versione precedente ...
        print("\n--- Risultati Finali di Validazione ---")
        for name, acc in self.final_accuracies.items():
            print(f"{name}: {acc:.4f}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        for name, h in self.all_histories.items():
            val_loss = h['val_loss']; val_acc = h['val_accuracy']
            train_loss = h['loss']; train_acc = h['accuracy']
            epochs_range = range(len(val_loss))
            
            ax1.plot(epochs_range, train_loss, label=f"{name} Train Loss")
            ax1.plot(epochs_range, val_loss, '--', label=f"{name} Val Loss")
            ax2.plot(epochs_range, train_acc, label=f"{name} Train Accuracy")
            ax2.plot(epochs_range, val_acc, '--', label=f"{name} Val Accuracy")

        ax1.set(title='Loss', xlabel='Epoca', ylabel='Loss')
        ax1.legend(); ax1.grid()
        ax2.set(title='Accuratezza', xlabel='Epoca', ylabel='Accuratezza')
        ax2.legend(); ax2.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'training_results.png'))
        plt.show() if show_plots else plt.close()


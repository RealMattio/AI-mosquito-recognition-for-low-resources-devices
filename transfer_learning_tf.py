import tensorflow as tf
from keras import layers, models, applications, optimizers
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import numpy as np
import json
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold # <-- IMPORTANTE: Per la K-Fold

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
                 early_stop_patience:int=10, 
                 # --- NUOVI PARAMETRI ---
                 k_folds:int=5,
                 lr_patience:int=3,
                 # ---------------------
                 models_dir:str='keras_models',
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
        # --- NUOVI ATTRIBUTI ---
        self.k_folds = k_folds
        self.lr_patience = lr_patience
        # -----------------------
        self.models_dir = models_dir
        self.results_dir = results_dir
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.img_height = 224
        self.img_width = 224
        
        # --- ATTRIBUTI PER DATI E RISULTATI MODIFICATI ---
        self.all_filepaths = []
        self.all_labels = []
        self.test_ds = None
        self.class_names = []
        self.all_cv_histories = {} # Salva le medie delle storie CV per modello
        self.final_model_histories = {} # Salva la storia dell'addestramento finale
        self.final_accuracies = {}
        
    def _load_filepaths_and_labels(self, data_directory):
        """
        Funzione helper per caricare i percorsi dei file e le etichette da una directory.
        """
        filepaths = []
        labels = []
        
        # Assumiamo che le sottocartelle siano le classi
        class_dirs = sorted([d for d in os.scandir(data_directory) if d.is_dir()], key=lambda d: d.name)
        self.class_names = [d.name for d in class_dirs]
        label_map = {name: i for i, name in enumerate(self.class_names)}

        for class_dir in class_dirs:
            for file in os.scandir(class_dir.path):
                if file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    filepaths.append(file.path)
                    labels.append(label_map[class_dir.name])
        
        return filepaths, labels
    
    def _parse_image(self, filename, label):
        """
        Funzione per leggere e decodificare un'immagine da un percorso file.
        """
        image = tf.io.read_file(filename)
        image = tf.io.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, [self.img_height, self.img_width])
        return image, label

    def _create_dataset_from_slices(self, filepaths, labels, shuffle=False):
        """
        Crea un tf.data.Dataset da slice di percorsi e etichette.
        """
        dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(filepaths))
        dataset = dataset.map(self._parse_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    def prepare_data(self):
        """
        MODIFICATO: Carica TUTTI i dati di training/validazione per la K-Fold
        e prepara il test set separatamente.
        """
        print("--- Fase 1: Caricamento e combinazione dati per Cross-Validation ---")
        
        # Combina dati da train_dir e val_dir
        train_filepaths, train_labels = self._load_filepaths_and_labels(self.train_dir)
        val_filepaths, val_labels = self._load_filepaths_and_labels(self.val_dir)
        
        self.all_filepaths = np.array(train_filepaths + val_filepaths)
        self.all_labels = np.array(train_labels + val_labels)
        
        print(f"Trovate {len(self.all_filepaths)} immagini totali per training/validazione.")
        print(f"Classi trovate: {self.class_names}")

        # Prepara il test set (rimane invariato)
        if os.path.exists(self.test_dir):
            print("\n--- Caricamento dati di Test ---")
            self.test_ds = tf.keras.utils.image_dataset_from_directory(
                self.test_dir, label_mode='int', seed=123,
                image_size=(self.img_height, self.img_width), batch_size=self.batch_size, shuffle=False
            ).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
            print("Test set caricato.")


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
        # ... questo metodo è identico alla versione precedente ...
        print(f"\n--- Inizio Valutazione Completa di {model_name} sul Test Set ---")
        y_probs = model.predict(self.test_ds)
        y_pred = np.argmax(y_probs, axis=1)
        y_true = np.concatenate([y for x, y in self.test_ds], axis=0)
        results = {}
        print("\nClassification Report:")
        report_dict = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        report_str = classification_report(y_true, y_pred, target_names=self.class_names)
        print(report_str)
        results['classification_report'] = report_dict
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
            print("\nLa curva ROC per il multi-classe non è stata implementata in questo esempio.")
        json_path = os.path.join(self.results_dir, f'evaluation_metrics_{model_name}.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Metriche di valutazione complete salvate in: {json_path}")

    def run_transfer_learning(self):
        """
        NUOVA VERSIONE: Esegue il ciclo completo con K-Fold Cross-Validation,
        learning rate adattivo e addestramento finale.
        """
        self.prepare_data()
        
        skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=42)

        for model_name in self.models_names:
            print(f"\n===== INIZIO PROCESSO PER MODELLO: {model_name} =====")
            
            fold_histories = []
            fold_val_accuracies = []
            
            # --- FASE A: K-FOLD CROSS VALIDATION ---
            print(f"--- Avvio {self.k_folds}-Fold Cross-Validation ---")
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(self.all_filepaths, self.all_labels)):
                print(f"\n--- Fold {fold + 1}/{self.k_folds} ---")
                
                # 1. Suddivisione dati per il fold corrente
                train_filepaths, val_filepaths = self.all_filepaths[train_idx], self.all_filepaths[val_idx]
                train_labels, val_labels = self.all_labels[train_idx], self.all_labels[val_idx]

                train_ds = self._create_dataset_from_slices(train_filepaths, train_labels, shuffle=True)
                val_ds = self._create_dataset_from_slices(val_filepaths, val_labels)

                # 2. Creazione e compilazione del modello (DA ZERO per ogni fold)
                model = self.get_model(model_name)
                model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate),
                              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                
                # 3. Definizione dei Callbacks con LR ADATTIVO
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_accuracy', 
                        patience=self.early_stop_patience, 
                        restore_best_weights=True,
                        verbose=1
                    ),
                    # --- LEARNING RATE ADATTIVO ---
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss', # Monitora la loss di validazione
                        factor=0.2,         # Riduci LR del 80% (1-0.2)
                        patience=self.lr_patience,  
                        min_lr=1e-6,        # LR minimo
                        verbose=1
                    )
                ]

                # 4. Addestramento sul fold
                history = model.fit(
                    train_ds, epochs=self.num_epochs, validation_data=val_ds, callbacks=callbacks, verbose=2)
                
                # 5. Salvataggio risultati del fold
                fold_histories.append(history.history)
                best_val_acc = max(history.history['val_accuracy'])
                fold_val_accuracies.append(best_val_acc)
                print(f"Fold {fold + 1} - Miglior val_accuracy: {best_val_acc:.4f}")

            # Calcolo e stampa delle performance medie della CV
            mean_cv_accuracy = np.mean(fold_val_accuracies)
            std_cv_accuracy = np.std(fold_val_accuracies)
            self.final_accuracies[f"{model_name}_CV"] = f"{mean_cv_accuracy:.4f} +/- {std_cv_accuracy:.4f}"
            print(f"\n--- Risultato Cross-Validation per {model_name} ---")
            print(f"Accuratezza media sui {self.k_folds} folds: {mean_cv_accuracy:.4f} (std: {std_cv_accuracy:.4f})")

            # --- FASE B: ADDESTRAMENTO FINALE SUL DATASET COMPLETO ---
            print(f"\n--- Avvio Addestramento Finale di {model_name} su tutti i dati ---")
            
            # 1. Creazione dataset con tutti i dati (train + val)
            full_train_ds = self._create_dataset_from_slices(self.all_filepaths, self.all_labels, shuffle=True)
            
            # 2. Creazione e compilazione del modello finale
            final_model = self.get_model(model_name)
            final_model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate),
                                loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
            # 3. Addestramento (senza validation data e early stopping basato su val_)
            #    Potremmo addestrare per un numero fisso di epoche basato sulla media CV
            #    o usare EarlyStopping su 'loss'
            since = time.time()
            history_final = final_model.fit(full_train_ds, epochs=self.num_epochs, verbose=1)
            time_elapsed = time.time() - since
            print(f"Addestramento finale completato in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
            
            self.final_model_histories[model_name] = history_final.history
            
            # 4. Salvataggio del modello finale
            filename = f"{model_name}_final_model.keras"
            path = os.path.join(self.models_dir, filename)
            final_model.save(path)
            print(f"Modello finale salvato in: {path}")

            # --- FASE C: VALUTAZIONE FINALE SUL TEST SET ---
            if self.test_ds:
                self.evaluate_model(final_model, model_name)


    def save_training_results(self, show_plots=False):
        # ... questo metodo può essere adattato per visualizzare le medie della CV
        # ... o le curve di training del modello finale. Qui visualizzo quelle finali.
        print("\n--- Risultati Finali di Validazione (dalla Cross-Validation) ---")
        for name, acc in self.final_accuracies.items():
            print(f"{name}: {acc}")
        
        # Grafici basati sull'addestramento del modello finale
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        for name, h in self.final_model_histories.items():
            train_loss = h['loss']
            train_acc = h['accuracy']
            epochs_range = range(len(train_loss))
            
            # Plot Loss
            ax_loss = ax
            ax_loss.plot(epochs_range, train_loss, label=f"{name} Train Loss")
            ax_loss.set_xlabel('Epoca', color='black')
            ax_loss.set_ylabel('Loss', color='blue')
            ax_loss.tick_params(axis='y', labelcolor='blue')
            
            # Plot Accuracy
            ax_acc = ax_loss.twinx()
            ax_acc.plot(epochs_range, train_acc, '--', label=f"{name} Train Accuracy")
            ax_acc.set_ylabel('Accuratezza', color='green')
            ax_acc.tick_params(axis='y', labelcolor='green')

        ax.set_title('Loss e Accuratezza del Training Finale')
        fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax.transAxes)
        ax.grid()
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'final_training_results.png'))
        plt.show() if show_plots else plt.close()
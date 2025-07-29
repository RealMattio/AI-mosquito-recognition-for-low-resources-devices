import tensorflow as tf
from keras import layers, models, applications, optimizers # Corretto l'import per Keras 3
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import numpy as np
import json
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import class_weight
from noDense_model import ModelFactory
from keras.metrics import Precision, Recall, AUC

# Assicuriamoci che la GPU venga usata correttamente
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU trovata e Memory Growth abilitato per: {gpus}")
    except RuntimeError as e:
        print(f"Errore nell'impostare la Memory Growth: {e}")

class TransferLearning:
    # --- MODIFICA: Aggiunto il parametro image_size ---
    def __init__(self, train_dir, val_dir, test_dir,
                 image_size: tuple = (224, 224), # <-- NUOVO PARAMETRO
                 num_classes:int=2, batch_size=32, num_epochs=25,
                 learning_rate=0.001, models_names:list[str]=None,
                 early_stop_patience:int=10, 
                 k_folds:int=5,
                 lr_patience:int=3,
                 models_dir:str='keras_models',
                 results_dir:str='keras_models_performances',
                 mobilenet_alpha: float = 1.0,
                 name_to_save_models:list[str] = None):
        
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.models_names = models_names or ['ResNet50', 'MobileNetV2', 'NASNetMobile', 'CustomCNN_Conv1D_Classifier', 'CustomCNN_Conv2D_Classifier', 
                                             'MobileNet']
        self.early_stop_patience = early_stop_patience
        self.k_folds = k_folds
        self.lr_patience = lr_patience
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.mobilenet_alpha = mobilenet_alpha
        self.name_to_save_models = name_to_save_models or [f"ResNet50_{time.strftime('%Y%m%d_%H%M%S')}", f"MobileNetV2_{time.strftime('%Y%m%d_%H%M%S')}", f"NASNetMobile_{time.strftime('%Y%m%d_%H%M%S')}", 
                                                           f"CustomCNN_Conv1D_{time.strftime('%Y%m%d_%H%M%S')}", f"CustomCNN_Conv2D_{time.strftime('%Y%m%d_%H%M%S')}", f"MobileNet_{time.strftime('%Y%m%d_%H%M%S')}"]
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        if len(self.models_names) != len(self.name_to_save_models):
            raise ValueError("models_names e name_to_save_models devono avere la stessa lunghezza.")
        
        # --- MODIFICA: Imposta le dimensioni dell'immagine dai parametri ---
        if not isinstance(image_size, tuple) or len(image_size) != 2:
            raise ValueError("image_size deve essere una tupla di due interi, es. (224, 224)")
        self.image_size = image_size
        self.img_height = self.image_size[0]
        self.img_width = self.image_size[1]
        
        self.all_filepaths = []
        self.all_labels = []
        self.test_ds = None
        self.class_names = []
        self.final_model_histories = {}
        self.final_accuracies = {}
    
    def _load_filepaths_and_labels(self, data_directory):
        filepaths, labels = [], []
        class_dirs = sorted([d for d in os.scandir(data_directory) if d.is_dir()], key=lambda d: d.name)
        if not self.class_names:
            self.class_names = [d.name for d in class_dirs]
        label_map = {name: i for i, name in enumerate(self.class_names)}

        for class_dir in class_dirs:
            for file in os.scandir(class_dir.path):
                if file.name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    filepaths.append(file.path)
                    labels.append(label_map[class_dir.name])
        return filepaths, labels
    
    def _parse_image(self, filename, label):
        image = tf.io.read_file(filename)
        image = tf.io.decode_jpeg(image, channels=3)
        # Questa funzione ora usa le dimensioni configurabili
        image = tf.image.resize(image, [self.img_height, self.img_width])
        return image, label

    def _create_dataset_from_slices(self, filepaths, labels, shuffle=False):
        dataset = tf.data.Dataset.from_tensor_slices((filepaths, labels))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(filepaths), reshuffle_each_iteration=True)
        dataset = dataset.map(self._parse_image, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    def prepare_data(self):
        # ... (codice invariato) ...
        print("--- Fase 1: Caricamento e combinazione dati per Cross-Validation ---")
        train_filepaths, train_labels = self._load_filepaths_and_labels(self.train_dir)
        val_filepaths, val_labels = self._load_filepaths_and_labels(self.val_dir)
        self.all_filepaths = np.array(train_filepaths + val_filepaths)
        self.all_labels = np.array(train_labels + val_labels)
        print(f"Trovate {len(self.all_filepaths)} immagini totali per training/validazione.")
        print(f"Classi trovate: {self.class_names}")
        if os.path.exists(self.test_dir) and any(os.scandir(self.test_dir)):
            print("\n--- Caricamento dati di Test ---")
            test_filepaths, test_labels = self._load_filepaths_and_labels(self.test_dir)
            self.test_ds = self._create_dataset_from_slices(test_filepaths, test_labels, shuffle=False)
            print("Test set caricato.")
        else:
            print("\nCartella di test non trovata o vuota. La valutazione finale sarà saltata.")
    
    
    def build_modular_cnn(self):
        """
        Costruisce una CNN modulare che si adatta a diverse dimensioni di input.
        """
        input_shape = (self.img_height, self.img_width, 3)  # Assicurati che l'input sia in formato (H, W, C)
        # 1. Definisci l'input del modello
        inputs = layers.Input(shape=input_shape)

        # 2. Costruisci la base convoluzionale
        x = layers.Rescaling(1./255)(inputs)
        x = layers.Conv2D(8, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Conv2D(16, (3, 3), activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        # L'output di questa base avrà una forma (H, W, C) che dipende dall'input
        conv_base_output = layers.Conv2D(32, (3, 3), activation='relu')(x)

        # 3. Calcola dinamicamente la dimensione per il flatten
        # Otteniamo la forma dell'output della base (es. (None, 10, 10, 64))
        shape = conv_base_output.shape
        # Calcoliamo la dimensione del vettore appiattito (es. 10 * 10 * 64 = 6400)
        flattened_size = shape[1] * shape[2] * shape[3]
        
        # 4. Costruisci il classificatore dinamico
        # Il Reshape ora usa la dimensione calcolata al volo
        x = layers.Reshape((flattened_size, 1))(conv_base_output)

        # Il kernel della Conv1D ora usa la dimensione calcolata al volo
        x = layers.Conv1D(filters=32, kernel_size=flattened_size, activation='relu', padding='valid')(x)

        # Il resto del classificatore
        x = layers.Reshape((32, 1))(x) # Questo 64 è fisso perché è il n° di filtri del layer precedente
        x = layers.Conv1D(self.num_classes, 32, activation='softmax', padding='valid')(x)

        # Flatten finale per ottenere l'output nella forma corretta (batch, num_classes)
        outputs = layers.Flatten()(x)

        # 5. Crea e restituisci il modello finale
        model = models.Model(inputs=inputs, outputs=outputs)
        return model

    def get_model(self, model_name_str):
        '''
        input_shape = (self.img_height, self.img_width, 3)
        
        # --- NUOVO: Blocco per il modello Custom CNN ---
        if model_name_str == 'CustomCNN':
            print("Costruzione del modello CustomCNN from scratch...")
            model = self.build_modular_cnn()
            return model

        # La logica per i modelli di transfer learning rimane la stessa
        inputs = tf.keras.Input(shape=input_shape)

        data_augmentation = models.Sequential([
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
            if self.img_height < 32 or self.img_width < 32:
                raise ValueError("NASNetMobile richiede una dimensione di input di almeno 32x32.")
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
        
        return models.Model(inputs, outputs)
        '''
        model = ModelFactory.create_model(model_name_str, (self.img_height, self.img_width, 3), self.num_classes)
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
        self.prepare_data()
        skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=42)

        for model_name, name_to_save in zip(self.models_names, self.name_to_save_models):
            print(f"\n===== INIZIO PROCESSO PER MODELLO: {model_name} =====")
            
            fold_val_accuracies, fold_epochs = [], []
            
            print(f"--- Avvio {self.k_folds}-Fold Cross-Validation ---")
            for fold, (train_idx, val_idx) in enumerate(skf.split(self.all_filepaths, self.all_labels)):
                print(f"\n--- Fold {fold + 1}/{self.k_folds} ---")
                
                train_ds = self._create_dataset_from_slices(self.all_filepaths[train_idx], self.all_labels[train_idx], shuffle=True)
                val_ds = self._create_dataset_from_slices(self.all_filepaths[val_idx], self.all_labels[val_idx])

                # <-- MODIFICA: Calcolo dei pesi per il fold di training corrente -->
                print("Calcolo dei pesi per il bilanciamento del dataset di training del fold...")
                train_labels_fold = self.all_labels[train_idx]
                class_labels = np.unique(train_labels_fold)
                weights = class_weight.compute_class_weight(
                    'balanced',
                    classes=class_labels,
                    y=train_labels_fold
                )
                class_weights_fold = dict(zip(class_labels, weights))
                print(f"Pesi calcolati per il fold: {class_weights_fold}")
                # <-- FINE MODIFICA -->
                
                model = self.get_model(model_name)
                model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate),
                            loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), 
                                                                 tf.keras.metrics.Recall(name='recall'), tf.keras.metrics.AUC(name='auc')])
                
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=self.early_stop_patience, restore_best_weights=True, verbose=1),
                    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=self.lr_patience, min_lr=1e-6, verbose=1)
                ]
                
                # <-- MODIFICA: Passaggio dei pesi al metodo fit() -->
                history = model.fit(train_ds, epochs=self.num_epochs, validation_data=val_ds, callbacks=callbacks,
                                    class_weight=class_weights_fold, verbose=1)
                # <-- FINE MODIFICA -->
                
                fold_val_accuracies.append(max(history.history['val_accuracy']))
                fold_epochs.append(len(history.history['val_loss']))
                print(f"Fold {fold + 1} - Miglior val_accuracy: {fold_val_accuracies[-1]:.4f} in {fold_epochs[-1]} epoche")

            mean_cv_accuracy, std_cv_accuracy = np.mean(fold_val_accuracies), np.std(fold_val_accuracies)
            optimal_epochs = int(np.mean(fold_epochs))
            self.final_accuracies[f"{model_name}_CV"] = f"{mean_cv_accuracy:.4f} +/- {std_cv_accuracy:.4f}"
            print(f"\n--- Risultato Cross-Validation per {model_name} ---")
            print(f"Accuratezza media sui {self.k_folds} folds: {mean_cv_accuracy:.4f} (std: {std_cv_accuracy:.4f})")
            print(f"Numero ottimale di epoche suggerito: {optimal_epochs}")

            print(f"\n--- Avvio Addestramento Finale di {model_name} su tutti i dati ---")
            full_train_ds = self._create_dataset_from_slices(self.all_filepaths, self.all_labels, shuffle=True)
            
            # <-- MODIFICA: Calcolo dei pesi per l'intero dataset di training -->
            print("Calcolo dei pesi per il bilanciamento del dataset completo...")
            class_labels_final = np.unique(self.all_labels)
            weights_final = class_weight.compute_class_weight(
                'balanced',
                classes=class_labels_final,
                y=self.all_labels
            )
            class_weights_final = dict(zip(class_labels_final, weights_final))
            print(f"Pesi calcolati per il training finale: {class_weights_final}")
            # <-- FINE MODIFICA -->
            
            final_model = self.get_model(model_name)
            final_model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate),
                            loss='binary_crossentropy', metrics=['accuracy'])
            
            # <-- MODIFICA: Passaggio dei pesi al metodo fit() finale -->
            history_final = final_model.fit(full_train_ds, epochs=optimal_epochs, 
                                            class_weight=class_weights_final, verbose=1)
            # <-- FINE MODIFICA -->
            
            self.final_model_histories[model_name] = history_final.history
            
            filename = f"{name_to_save}_final_model.keras"
            path = os.path.join(self.models_dir, filename)
            final_model.save(path)
            print(f"Modello finale salvato in: {path}")

            # --- Salvataggio della storia del modello in un file JSON ---
            history_dict = history_final.history
            # Converti i valori in float standard di Python per la compatibilità JSON
            for key in history_dict:
                history_dict[key] = [float(val) for val in history_dict[key]]
                
            history_path = os.path.join(self.results_dir, f"history_{name_to_save}.json")
            with open(history_path, 'w') as f:
                json.dump(history_dict, f, indent=4)
            print(f"Storia dell'addestramento salvata in: {history_path}")

            if self.test_ds:
                self.evaluate_model(final_model, name_to_save) # <-- Assicurati che il codice di evaluate sia qui


    def save_training_results(self, output_filename=f'final_training_results.png', show_plots=False):
        print("\n--- Risultati Finali di Validazione (dalla Cross-Validation) ---")
        for name, acc in self.final_accuracies.items():
            print(f"{name}: {acc}")
        
        # Grafici basati sull'addestramento del modello finale
        fig, ax_loss = plt.subplots(1, 1, figsize=(14, 8)) # Leggermente più grande per una migliore leggibilità
        ax_acc = ax_loss.twinx() # Crea l'asse per l'accuracy una sola volta
        
        # --- MODIFICA: Creazione di una palette di colori ---
        # Usiamo una palette di colori standard di Matplotlib ('tab10' ha 10 colori distinti)
        # per assegnare un colore diverso a ogni modello.
        colors = plt.cm.get_cmap('tab10').colors
        
        # Impostazioni generali degli assi (fuori dal ciclo)
        ax_loss.set_xlabel('Epoca', fontsize=12)
        ax_loss.set_ylabel('Loss', fontsize=12)
        ax_acc.set_ylabel('Accuratezza', fontsize=12)
        ax_loss.set_title('Loss e Accuratezza del Training Finale per Modello', fontsize=16)
        ax_loss.grid(True, which='both', linestyle='--', linewidth=0.5)

        # Ciclo sui modelli, usando enumerate per ottenere un indice per i colori
        for i, (name, h) in enumerate(self.final_model_histories.items()):
            # Seleziona un colore dalla palette in modo ciclico
            color = colors[i % len(colors)]
            
            epochs_range = range(len(h['loss']))
            
            # --- MODIFICA: Uso di un colore diverso per ogni modello ---
            # Usiamo lo stesso colore per loss e accuracy di un modello, ma con stili diversi.
            
            # Plot Loss (linea continua)
            ax_loss.plot(epochs_range, h['loss'], color=color, linestyle='-', label=f"{name} Train Loss")
            
            # Plot Accuracy (linea tratteggiata)
            ax_acc.plot(epochs_range, h['accuracy'], color=color, linestyle='--', label=f"{name} Train Accuracy")

        # --- MODIFICA: Gestione della legenda migliorata per assi doppi ---
        # Raccogliamo le "handles" (le linee) e le etichette da entrambi gli assi
        handles_loss, labels_loss = ax_loss.get_legend_handles_labels()
        handles_acc, labels_acc = ax_acc.get_legend_handles_labels()
        
        # Combiniamo e mostriamo una singola legenda
        ax_loss.legend(handles_loss + handles_acc, labels_loss + labels_acc, loc='best')
        
        plt.tight_layout()
        # Salva il grafico
        output_path = os.path.join(self.results_dir, output_filename)
        plt.savefig(output_path)
        print(f"\nGrafico dei risultati di training salvato in: {output_path}")

        if show_plots:
            plt.show()
        else:
            plt.close()


def plot_saved_histories(results_dir, output_filename='final_training_results_from_json.png', show_plots=True):
    """
    Carica le storie di addestramento salvate come file JSON e genera i grafici di loss/accuratezza.
    
    Args:
        results_dir (str): La cartella dove sono stati salvati i file history_...json.
        output_filename (str): Il nome del file per il grafico generato.
        show_plots (bool): Se mostrare il grafico a schermo dopo averlo salvato.
    """
    print(f"Ricerca di file di history nella cartella: {results_dir}")
    
    all_histories = {}
    
    # Cerca e carica tutti i file di history
    for filename in os.listdir(results_dir):
        if filename.startswith('history_') and filename.endswith('.json'):
            # Estrai il nome del modello dal nome del file
            # Es: "history_ResNet50.json" -> "ResNet50"
            model_name = filename.replace('history_', '').replace('.json', '')
            
            filepath = os.path.join(results_dir, filename)
            with open(filepath, 'r') as f:
                history_data = json.load(f)
                all_histories[model_name] = history_data
                print(f"Caricata history per il modello: {model_name}")

    if not all_histories:
        print("Nessun file di history trovato. Impossibile generare il grafico.")
        return
    
    # Plot singolo di tutte le storie
    for name, history in all_histories.items():
        if 'loss' not in history or 'accuracy' not in history:
            print(f"ERRORE: Il file {filename} non contiene le chiavi 'loss' o 'accuracy'.")
            continue
        fig, ax_loss = plt.subplots(1, 1, figsize=(14, 8))
        ax_acc = ax_loss.twinx()
        epochs_range = range(len(history['loss']))
        ax_loss.set_xlabel('Epoca', fontsize=12)
        ax_loss.set_ylabel('Loss', fontsize=12)
        ax_acc.set_ylabel('Accuratezza', fontsize=12)
        ax_loss.set_title(f'Loss e Accuratezza per {name}', fontsize=16)
        ax_loss.grid(True, which='both', linestyle='--', linewidth=0.5)

        ax_loss.plot(epochs_range, history['loss'], label=f"{name} Train Loss", linestyle='-', color='red')
        ax_acc.plot(epochs_range, history['accuracy'], label=f"{name} Train Accuracy", linestyle='--', color='blue')
        ax_loss.legend(loc='best')
        plt.tight_layout()
        output_path = os.path.join(results_dir, f'history_plot_training_{name}.png')
        plt.savefig(output_path)
        print(f"Grafico della history di {name} salvato in: {output_path}")

    # --- La logica di plotting è la STESSA che hai già corretto ---
    fig, ax_loss = plt.subplots(1, 1, figsize=(14, 8))
    ax_acc = ax_loss.twinx()
    
    colors = plt.cm.get_cmap('tab10').colors
    
    ax_loss.set_xlabel('Epoca', fontsize=12)
    ax_loss.set_ylabel('Loss', fontsize=12)
    ax_acc.set_ylabel('Accuratezza', fontsize=12)
    ax_loss.set_title('Loss e Accuratezza del Training Finale (da file salvati)', fontsize=16)
    ax_loss.grid(True, which='both', linestyle='--', linewidth=0.5)

    for i, (name, h) in enumerate(all_histories.items()):
        color = colors[i % len(colors)]
        epochs_range = range(len(h['loss']))
        
        ax_loss.plot(epochs_range, h['loss'], color=color, linestyle='-', label=f"{name} Train Loss")
        ax_acc.plot(epochs_range, h['accuracy'], color=color, linestyle='--', label=f"{name} Train Accuracy")

    handles_loss, labels_loss = ax_loss.get_legend_handles_labels()
    handles_acc, labels_acc = ax_acc.get_legend_handles_labels()
    
    ax_loss.legend(handles_loss + handles_acc, labels_loss + labels_acc, loc='best')
    
    plt.tight_layout()
    output_path = os.path.join(results_dir, output_filename)
    plt.savefig(output_path)
    print(f"\nGrafico generato e salvato in: {output_path}")

    if show_plots:
        plt.show()
    else:
        plt.close()
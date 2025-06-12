from tensorflow.keras import layers, models, applications, optimizers
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os
import numpy as np
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

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
        self.models_names = models_names or ['ResNet50', 'MobileNetV2', 'NASNetMobile'] # ResNet18 non è in tf.keras.applications
        self.need_resize = need_resize
        self.need_normalize = need_normalize # La normalizzazione è gestita dai modelli Keras
        self.early_stop_patience = early_stop_patience
        self.models_dir = models_dir
        os.makedirs(self.models_dir, exist_ok=True)
        print(f"Utilizzo di TensorFlow. Dispositivi disponibili: {tf.config.list_physical_devices()}")

    def _preprocess_data(self, image, label):
        """Funzione di pre-processing per le immagini."""
        if self.need_resize:
            image = tf.image.resize(image, [224, 224])
        # La normalizzazione specifica di ImageNet è gestita dalle funzioni di preprocess_input
        # dei rispettivi modelli Keras.
        return image, label

    def prepare_data(self):
        """Prepara i dataset di training e validazione usando tf.data."""
        # Creazione dei dataset da tensori
        train_dataset = tf.data.Dataset.from_tensor_slices((self.X_train, self.y_train))
        val_dataset = tf.data.Dataset.from_tensor_slices((self.X_val, self.y_val))

        # Data augmentation solo per il training set
        data_augmentation = tf.keras.Sequential([
            layers.RandomFlip("horizontal"),
        ])

        # Applicazione delle trasformazioni e batching
        self.dataloaders = {}
        self.dataloaders['train'] = (
            train_dataset
            .shuffle(buffer_size=len(self.X_train))
            .map(self._preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.batch_size)
            .map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        self.dataloaders['val'] = (
            val_dataset
            .map(self._preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.batch_size)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
        )
        self.dataset_sizes = {'train': len(self.X_train), 'val': len(self.X_val)}
        print(f"Dataset sizes: {self.dataset_sizes}")


    def get_model(self, model_name_str, num_classes_val):
        """Crea un modello pre-addestrato con un nuovo classificatore."""
        if model_name_str == 'ResNet50':
            base_model = applications.ResNet50(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
            preprocess_input = applications.resnet50.preprocess_input
        elif model_name_str == 'MobileNetV2':
            base_model = applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
            preprocess_input = applications.mobilenet_v2.preprocess_input
        elif model_name_str == 'NASNetMobile':
            base_model = applications.NASNetMobile(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
            preprocess_input = applications.nasnet.preprocess_input
        else:
            raise ValueError("Model name non valido")

        base_model.trainable = False  # Congela i pesi del modello base

        inputs = tf.keras.Input(shape=(224, 224, 3))
        x = preprocess_input(inputs) # Applica la normalizzazione specifica del modello
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(num_classes_val, activation='softmax' if num_classes_val > 1 else 'sigmoid')(x)
        
        model = models.Model(inputs, outputs)
        return model

    def train_model_instance(self, model, model_name):
        """Compila e addestra il modello usando .fit() e callbacks."""
        # Callbacks per l'early stopping e il salvataggio del modello migliore
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=self.early_stop_patience, restore_best_weights=True)
        ]

        model.compile(optimizer=optimizers.Adam(learning_rate=self.learning_rate),
                      loss='sparse_categorical_crossentropy', # Ideale per etichette intere
                      metrics=['accuracy'])

        since = time.time()
        history = model.fit(
            self.dataloaders['train'],
            epochs=self.num_epochs,
            validation_data=self.dataloaders['val'],
            callbacks=callbacks
        )
        time_elapsed = time.time() - since

        print(f"Training completed in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
        
        best_acc = max(history.history['val_accuracy'])
        # I pesi migliori sono già stati ripristinati da EarlyStopping
        return model, history.history, best_acc

    def save_model(self, model, model_name, accuracy):
        """Salva il modello Keras."""
        filename = f"{model_name}_{accuracy:.4f}.keras"
        path = os.path.join(self.models_dir, filename)
        model.save(path)
        print(f"Model saved to {path}")

    def run_transfer_learning(self):
        """Esegue il ciclo completo di addestramento per tutti i modelli specificati."""
        self.prepare_data()
        self.all_histories = {}
        self.final_accuracies = {}

        for model_name in self.models_names:
            print(f"\nTraining {model_name}")
            model = self.get_model(model_name, self.num_classes)
            trained_model, history, best_acc = self.train_model_instance(model, model_name)
            self.all_histories[model_name] = {
                'train_loss': history['loss'], 'val_loss': history['val_loss'],
                'train_acc': history['accuracy'], 'val_acc': history['val_accuracy']
            }
            self.final_accuracies[model_name] = best_acc
            self.save_model(trained_model, model_name, best_acc)

    def show_training_results(self, show_plots:bool=False):
        """Mostra grafici di loss e accuratezza per i modelli addestrati."""
        print("\n--- Risultati Finali ---")
        for name, acc in self.final_accuracies.items():
            print(f"{name}: {acc:.4f}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        for name, h in self.all_histories.items():
            epochs_i = range(1, len(h['train_loss']) + 1)
            ax1.plot(epochs_i, h['train_loss'], label=f"{name} train")
            ax1.plot(epochs_i, h['val_loss'], '--', label=f"{name} val")
            ax2.plot(epochs_i, h['train_acc'], label=f"{name} train")
            ax2.plot(epochs_i, h['val_acc'], '--', label=f"{name} val")
        
        ax1.set(title='Loss', xlabel='Epoca', ylabel='Loss')
        ax1.legend(); ax1.grid()
        ax2.set(title='Accuratezza', xlabel='Epoca', ylabel='Accuratezza')
        ax2.legend(); ax2.grid()
        plt.savefig('training_results_tf.png')
        plt.tight_layout()
        if show_plots:
            plt.show()


def evaluate_and_save_results(X_test, y_test, models_dir='saved_models', num_classes=2,
                              batch_size=32, need_resize=True,
                              output_json='test_results_tf.json', roc_plot_path='roc_curves_tf.png'):
    """
    Carica tutti i modelli TensorFlow salvati, li valuta sul test set,
    e salva metriche e curve ROC.
    """
    
    # Preparazione del dataset di test
    def preprocess_test(image, label):
        if need_resize:
            image = tf.image.resize(image, [224, 224])
        return image, label

    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test_loader = test_dataset.map(preprocess_test).batch(batch_size)

    results = {}
    all_fpr, all_tpr, all_auc = {}, {}, {}
    
    for filename in os.listdir(models_dir):
        if filename.endswith('.keras'):
            path = os.path.join(models_dir, filename)
            model_name = filename.replace('.keras', '')
            name = model_name.rsplit('_', 1)[0]
            
            try:
                # Carica il modello. La funzione di preprocessing è inclusa nel modello.
                model = models.load_model(path)
                print(f"Evaluating {name}...")

                # Ottieni le probabilità e le predizioni
                y_probs = model.predict(test_loader)
                y_pred = np.argmax(y_probs, axis=1)
                
                # Calcola le metriche
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

                model_results = {
                    'filename': filename,
                    'test_accuracy': acc, 'precision': prec, 'recall': rec, 'f1_score': f1
                }

                # Calcolo ROC e AUC
                y_true_bin = label_binarize(y_test, classes=np.arange(num_classes))
                if num_classes == 2:
                    fpr, tpr, _ = roc_curve(y_test, y_probs[:, 1])
                    roc_auc = auc(fpr, tpr)
                    model_results['roc_auc'] = roc_auc
                    all_fpr[name] = fpr
                    all_tpr[name] = tpr
                    all_auc[name] = roc_auc
                else: # Multi-class
                    roc_auc_dict = {}
                    for i in range(num_classes):
                        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
                        roc_auc_dict[i] = auc(fpr, tpr)
                    model_results['roc_auc'] = roc_auc_dict
                
                results[name] = model_results

            except Exception as e:
                print(f"Errore durante la valutazione di {filename}: {e}")

    # Salva i risultati in JSON
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Risultati di test salvati in {output_json}")

    # Salva le curve ROC (solo per classificazione binaria)
    if num_classes == 2 and all_fpr:
        plt.figure(figsize=(8, 6))
        for name in all_fpr:
            plt.plot(all_fpr[name], all_tpr[name], label=f"{name} (AUC = {all_auc[name]:.2f})")
        
        plt.plot([0, 1], [0, 1], 'k--', label='Chance')
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

# Esempio di utilizzo (commentato)
# if __name__ == '__main__':
#     # Carica i tuoi dati (esempio con dati casuali)
#     (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
#     X_train, X_val = X_train[:40000], X_train[40000:]
#     y_train, y_val = y_train[:40000], y_train[40000:]
#
#     # Riduci i dati per un test rapido
#     X_train, y_train = X_train[:1000], y_train[:1000]
#     X_val, y_val = X_val[:200], y_val[:200]
#     X_test, y_test = X_test[:200], y_test[:200]
#
#     # Esegui il training
#     tl_trainer = TransferLearning(
#         X_train, y_train.flatten(), X_val, y_val.flatten(),
#         num_classes=10,
#         num_epochs=5, # Riduci le epoche per un test rapido
#         early_stop_patience=3,
#         models_names=['MobileNetV2'] # Usa un solo modello per velocità
#     )
#     tl_trainer.run_transfer_learning()
#     tl_trainer.show_training_results()
#
#     # Esegui la valutazione
#     evaluate_and_save_results(X_test, y_test.flatten(), num_classes=10)
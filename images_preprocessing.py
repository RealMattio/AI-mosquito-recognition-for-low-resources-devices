import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
import cv2 # OpenCV per il processamento delle immagini
from PIL import Image # Pillow per un'alternativa al caricamento immagini

class ImagePreprocessor:
    """
    Classe per il preprocessing delle immagini in un dataset.
    Include caricamento, analisi, suddivisione in set di addestramento/validazione/test,
    e preprocessing (ridimensionamento, conversione in scala di grigi, normalizzazione).
    """

    def __init__(self, image_dir='./augmented_dataset', target_size=(128, 128), convert_to_grayscale=False,
                 test_set_size=0.10, validation_set_size=0.20, random_state=42):
        """
        Inizializza il preprocessore con i parametri specificati.
        :param image_dir: Directory contenente le immagini, strutturate per classi.
        :param target_size: Dimensione a cui ridimensionare le immagini (altezza, larghezza).
        :param convert_to_grayscale: Se True, converte le immagini in scala di grigi. Se il colore non è discriminante, impostalo a True.
        :param test_set_size: Proporzione del dataset da riservare al test set (0-1).
        :param validation_set_size: Proporzione del dataset rimanente da riservare al validation set (0-1).
        :param random_state: Seme per la riproducibilità della suddivisione casuale del dataset.
        """
        self.image_dir = image_dir
        self.target_size = target_size
        self.convert_to_grayscale = convert_to_grayscale
        self.test_set_size = test_set_size
        self.validation_set_size = validation_set_size
        self.random_state = random_state 

    
    def load_images_and_labels(self, image_dir):
        """
        Carica immagini e etichette da una directory strutturata per classi.
        Ogni sottocartella in image_dir è considerata una classe.
        """
        images = []
        labels = []
        class_names = sorted(os.listdir(image_dir))
        label_map = {class_name: i for i, class_name in enumerate(class_names)}

        print(f"Trovate {len(class_names)} classi: {class_names}")

        for class_name in class_names:
            class_path = os.path.join(image_dir, class_name)
            if os.path.isdir(class_path):
                for image_name in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_name)
                    try:
                        # Caricamento con OpenCV (puoi usare PIL se preferisci)
                        img = cv2.imread(image_path)
                        if img is not None:
                            images.append(img)
                            labels.append(label_map[class_name])
                        else:
                            print(f"Attenzione: Impossibile caricare l'immagine {image_path}")
                    except Exception as e:
                        print(f"Errore durante il caricamento di {image_path}: {e}")

        print(f"Caricate {len(images)} immagini.")
        return np.array(images, dtype=object), np.array(labels), class_names, label_map

    def analyze_dataset(self, labels, class_names):
        """
        Analizza la distribuzione delle classi.
        """
        counts = Counter(labels)
        print("\n--- Analisi del Dataset ---")
        print("Distribuzione delle immagini per classe:")
        for i, class_name in enumerate(class_names):
            print(f"- Classe '{class_name}' (ID: {i}): {counts[i]} immagini")

        # Verifica sbilanciamento
        if len(set(counts.values())) > 1: # Se ci sono conteggi diversi
            print("Attenzione: Il dataset potrebbe essere sbilanciato.")
            # Qui si potrebbero implementare tecniche di bilanciamento come oversampling, 
            # undersampling, o l'uso di pesi per le classi durante l'addestramento del modello
            
    def display_sample_images(self, images, labels, class_names, num_samples_per_class=3):
        """
        Visualizza alcuni campioni per ogni classe.
        """
        print("\n--- Campioni di Immagini per Classe ---")
        unique_labels = np.unique(labels)
        fig, axes = plt.subplots(len(unique_labels), num_samples_per_class, figsize=(10, len(unique_labels) * 2))
        fig.suptitle("Campioni di Immagini per Classe", fontsize=16)

        for i, label in enumerate(unique_labels):
            class_images = [img for img, lbl in zip(images, labels) if lbl == label]
            if not class_images:
                print(f"Nessuna immagine trovata per la classe {class_names[label]}")
                continue

            for j in range(num_samples_per_class):
                if j < len(class_images):
                    ax = axes[i, j] if len(unique_labels) > 1 else axes[j]
                    # OpenCV carica in BGR, Matplotlib si aspetta RGB
                    img = np.array(class_images[j], dtype=np.uint8)
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    ax.imshow(img_rgb)
                    ax.set_title(f"Classe: {class_names[label]}")
                    ax.axis('off')
                else:
                    # Nascondi subplot vuoti se ci sono meno campioni della num_samples_per_class
                    if len(unique_labels) > 1:
                        axes[i,j].axis('off')
                    else:
                        axes[j].axis('off')
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Aggiusta il layout per far spazio al titolo
        plt.show()


    def split_dataset(self, images, labels):
        """
        Suddivide il dataset in set di addestramento, validazione e test in modo stratificato.
        """
        print("\n--- Suddivisione del Dataset ---")
        # Prima suddivisione: training + validation vs test
        train_val_images, test_images, train_val_labels, test_labels = train_test_split(
            images, labels,
            test_size=self.test_set_size,
            stratify=labels, # Mantiene le proporzioni delle classi
            random_state=self.random_state
        )

        # Seconda suddivisione: training vs validation
        # Calcola la dimensione del set di validazione rispetto al set train_val combinato
        relative_validation_size = self.validation_set_size / (1 - self.test_set_size)
        train_images, val_images, train_labels, val_labels = train_test_split(
            train_val_images, train_val_labels,
            test_size=relative_validation_size,
            stratify=train_val_labels, # Mantiene le proporzioni delle classi
            random_state=self.random_state
        )

        print(f"Dimensioni dei set:")
        print(f"- Training:   {len(train_images)} immagini, {len(np.unique(train_labels))} classi")
        print(f"- Validation: {len(val_images)} immagini, {len(np.unique(val_labels))} classi")
        print(f"- Test:       {len(test_images)} immagini, {len(np.unique(test_labels))} classi")

        print("\nDistribuzione classi nel Training set:")
        print(Counter(train_labels))
        print("\nDistribuzione classi nel Validation set:")
        print(Counter(val_labels))
        print("\nDistribuzione classi nel Test set:")
        print(Counter(test_labels))

        return train_images, train_labels, val_images, val_labels, test_images, test_labels

    
    def preprocess_image(self, image, target_size, convert_to_grayscale, training_mean=None, training_std=None):
        """
        Esegue il preprocessing su una singola immagine:
        - Ridimensionamento
        - Conversione in scala di grigi (opzionale)
        - Normalizzazione
        """
        img = np.array(image, dtype=np.uint8)
        # Ridimensionamento
        processed_image = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)

        # Conversione in scala di grigi (opzionale)
        if convert_to_grayscale:
            if len(processed_image.shape) == 3 and processed_image.shape[2] == 3: # Se è a colori
                processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
            # Aggiungi una dimensione per il canale se è in scala di grigi e il modello se lo aspetta
            # (es. per CNN che si aspettano (altezza, larghezza, canali))
            if len(processed_image.shape) == 2:
                processed_image = np.expand_dims(processed_image, axis=-1)


        # Normalizzazione
        processed_image = processed_image.astype(np.float32)
        if training_mean is not None and training_std is not None:
            # Standardizzazione (Z-score normalization)
            processed_image = (processed_image - training_mean) / training_std
        else:
            # Scala i pixel a [0, 1]
            processed_image /= 255.0

        return processed_image

    def preprocess_dataset(self, images_list, target_size, convert_to_grayscale, fit_on_training=False, training_mean=None, training_std=None):
        """
        Applica il preprocessing a una lista di immagini.
        Se fit_on_training è True, calcola media e deviazione standard (per la standardizzazione)
        sul set fornito (tipicamente il training set) e li restituisce.
        Altrimenti, usa training_mean e training_std forniti.
        """
        processed_images = []

        if fit_on_training:
            # Calcola media e deviazione standard SUL TRAINING SET
            # Prima ridimensiona e converti in scala di grigi se necessario
            temp_images_for_stats = []
            for img in images_list:
                img_uint8 = np.array(img, dtype=np.uint8)
                resized_img = cv2.resize(img_uint8, target_size, interpolation=cv2.INTER_AREA)
                if convert_to_grayscale:
                    if len(resized_img.shape) == 3 and resized_img.shape[2] == 3:
                        resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
                    if len(resized_img.shape) == 2:
                        resized_img = np.expand_dims(resized_img, axis=-1)
                temp_images_for_stats.append(resized_img.astype(np.float32) / 255.0) # Normalizza a [0,1] prima di calcolare media/std

            stacked_images = np.stack(temp_images_for_stats)
            training_mean = np.mean(stacked_images, axis=(0, 1, 2)) # Media per canale se a colori, o globale se grigio
            training_std = np.std(stacked_images, axis=(0, 1, 2))
            print(f"\nCalcolati dal training set:")
            print(f"  Media per la standardizzazione: {training_mean}")
            print(f"  Deviazione standard per la standardizzazione: {training_std}")

        for img in images_list:
            processed_images.append(self.preprocess_image(img, target_size, convert_to_grayscale, training_mean, training_std))

        if fit_on_training:
            return np.array(processed_images), training_mean, training_std
        else:
            return np.array(processed_images)

    # --- Esecuzione del Flusso di Preprocessing ---
    def run_preprocessing_pipeline(self):
        """ Esegue l'intero flusso di preprocessing del dataset:
        1. Caricamento e Analisi del Dataset
        2. Suddivisione del Dataset in Training, Validation e Test
        3. Preprocessing delle Immagini (ridimensionamento, conversione in scala di grigi, normalizzazione)
        """
        # 1. Caricamento e Analisi
        images, labels, class_names, label_map = self.load_images_and_labels(self.image_dir)

        if len(images) == 0:
            print(f"Nessuna immagine caricata. Controlla il percorso '{self.image_dir}' e la struttura delle cartelle.")
        else:
            self.analyze_dataset(labels, class_names)
            #self.display_sample_images(images, labels, class_names, num_samples_per_class=3)

            # 2. Suddivisione del Dataset
            X_train_orig, y_train, X_val_orig, y_val, X_test_orig, y_test = self.split_dataset(images, labels)

            # 3. Preprocessing delle Immagini
            print(f"\n--- Preprocessing delle Immagini (Target Size: {self.target_size}) ---")
            print(f"Conversione in scala di grigi: {'Sì' if self.convert_to_grayscale else 'No'}")

            # Scegli se standardizzare o normalizzare a [0,1]
            # Per standardizzare, imposta use_standardization = True
            # Per normalizzare a [0,1], imposta use_standardization = False
            use_standardization = True # CAMBIA QUESTO PER TESTARE

            if use_standardization:
                print("Utilizzo della standardizzazione (Z-score).")
                X_train, train_mean, train_std = self.preprocess_dataset(
                    X_train_orig, self.target_size, self.convert_to_grayscale, fit_on_training=True
                )
                X_val = self.preprocess_dataset(
                    X_val_orig, self.target_size, self.convert_to_grayscale, training_mean=train_mean, training_std=train_std
                )
                X_test = self.preprocess_dataset(
                    X_test_orig, self.target_size, self.convert_to_grayscale, training_mean=train_mean, training_std=train_std
                )
            else:
                print("Utilizzo della normalizzazione a [0,1].")
                X_train, _, _ = self.preprocess_dataset( # Ignora mean e std restituiti
                    X_train_orig, self.target_size, self.convert_to_grayscale, fit_on_training=False # fit_on_training=False implica normalizzazione [0,1]
                )
                X_val = self.preprocess_dataset(
                    X_val_orig, self.target_size, self.convert_to_grayscale, fit_on_training=False
                )
                X_test = self.preprocess_dataset(
                    X_test_orig, self.target_size, self.convert_to_grayscale, fit_on_training=False
                )


            print(f"\nForma dei dati preprocessati:")
            print(f"- X_train: {X_train.shape}, y_train: {y_train.shape}")
            print(f"- X_val:   {X_val.shape}, y_val: {y_val.shape}")
            print(f"- X_test:  {X_test.shape}, y_test: {y_test.shape}")

            # Verifica i valori dei pixel dopo il preprocessing (dovrebbero essere normalizzati)
            print(f"\nValori min/max di un campione X_train dopo il preprocessing:")
            if len(X_train) > 0:
                sample_processed_image = X_train[0]
                print(f"  Min: {sample_processed_image.min()}, Max: {sample_processed_image.max()}")
                print(f"  Media: {sample_processed_image.mean()}, Deviazione Std: {sample_processed_image.std()}")
                print(f"  Tipo di dati: {sample_processed_image.dtype}")

                # Visualizza un campione preprocessato
                plt.figure(figsize=(4,4))
                plt.title("Campione Preprocessato (dal Training Set)")
                # Se è in scala di grigi e ha un canale, rimuovilo per imshow
                if sample_processed_image.shape[-1] == 1:
                    plt.imshow(sample_processed_image.squeeze(), cmap='gray')
                else:
                    # Se hai usato la standardizzazione, i valori potrebbero non essere più in [0,1]
                    # quindi potresti aver bisogno di clipparli o scalarli per una visualizzazione corretta.
                    # Per semplicità, se standardizzato, visualizziamo senza clipping,
                    # ma tieni presente che i colori potrebbero apparire strani.
                    img_to_show = sample_processed_image
                    if use_standardization: # Riporta approssimativamente a [0,1] per visualizzazione
                        img_to_show = (sample_processed_image * train_std) + train_mean
                        img_to_show = np.clip(img_to_show, 0, 1) # Clipa a [0,1] per evitare warning imshow

                    plt.imshow(img_to_show)
                plt.axis('off')
                plt.show()

            print("\nPreprocessing completato.")
            return X_train, y_train, X_val, y_val, X_test, y_test, class_names, label_map
            # Ora X_train, y_train, X_val, y_val, X_test, y_test sono pronti
            # per essere usati per addestrare un modello di classificazione.
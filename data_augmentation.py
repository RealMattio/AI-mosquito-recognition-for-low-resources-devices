import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt # Opzionale, per visualizzazione
import numpy as np
import os
import shutil # Per creare/pulire directory

# --- 1. Definizione dei Parametri ---
dataset_dir = './dataset'  # MODIFICA QUESTO PERCORSO
output_augmented_dir = './augmented_dataset' # MODIFICA QUESTO PERCORSO

image_height = 180
image_width = 180
batch_size = 1 # Processiamo un'immagine alla volta per salvarla individualmente
seed = 42
num_augmentations_per_image = 5 # Quante versioni augmentate generare per ogni immagine originale

# --- 2. Preparazione Directory di Output ---
if os.path.exists(output_augmented_dir):
    print(f"La directory di output '{output_augmented_dir}' esiste già.")
    # Puoi decidere se pulirla o aggiungere un timestamp al nome della nuova directory
    # Per questo esempio, la puliamo se esiste per evitare accumulo.
    # ATTENZIONE: questo cancellerà il contenuto della directory!
    # user_input = input(f"Vuoi cancellare e ricreare la directory '{output_augmented_dir}'? (s/N): ")
    # if user_input.lower() == 's':
    #     shutil.rmtree(output_augmented_dir)
    #     print(f"Directory '{output_augmented_dir}' cancellata.")
    # else:
    #     print("Operazione annullata. Modifica 'output_augmented_dir' o gestisci la directory esistente.")
    #     exit()

# Non è necessario ricreare la directory principale qui, lo faremo per le sottoclassi.
# os.makedirs(output_augmented_dir, exist_ok=True)
# print(f"Directory di output '{output_augmented_dir}' pronta.")


# --- 3. Caricamento del Dataset Completo ---
# Carichiamo l'intero dataset senza suddivisione in train/validation
# shuffle=False è importante per associare correttamente le immagini originali alle loro augmentations
# sebbene per il solo salvataggio potrebbe non essere strettamente necessario,
# è buona prassi se si volesse tracciare l'origine.
full_dataset = keras.utils.image_dataset_from_directory(
    dataset_dir,
    labels='inferred',
    label_mode='int', # Usiamo 'int' per ottenere indici di classe diretti
    image_size=(image_height, image_width),
    interpolation='nearest',
    batch_size=batch_size, # Carichiamo un'immagine alla volta
    shuffle=False, # Importante per mantenere l'ordine se necessario
    seed=seed
)

class_names = full_dataset.class_names
print("Nomi delle classi trovate:", class_names)

# Crea le directory di output per ogni classe
for class_name in class_names:
    class_output_dir = os.path.join(output_augmented_dir, class_name)
    os.makedirs(class_output_dir, exist_ok=True)
print(f"Directory di output per le classi create in '{output_augmented_dir}'.")


# --- 4. Data Augmentation Layers ---
# Definiamo i layer di data augmentation come nel codice precedente
data_augmentation_layers = keras.Sequential(
    [
        layers.RandomFlip("horizontal", seed=seed),
        layers.RandomRotation(0.2, seed=seed), # Aumentata un po' la rotazione per più varietà
        layers.RandomZoom(0.2, seed=seed),   # Aumentato un po' lo zoom
        layers.RandomContrast(0.2, seed=seed),
        layers.RandomBrightness(0.2, seed=seed) # Assicurati che la tua versione TF lo supporti
    ],
    name="data_augmentation",
)
# La data augmentation è una tecnica per aumentare artificialmente
# le dimensioni del training set creando versioni modificate delle immagini esistenti[cite: 2951, 2953].


# --- 5. Processo di Augmentation e Salvataggio ---
print("\nInizio processo di data augmentation e salvataggio...")

# Contatore per i nomi dei file originali (per tracciamento se necessario)
original_image_counter = 0

# Iteriamo su ogni batch del dataset originale (ogni batch contiene 1 immagine)
for images_batch, labels_batch in full_dataset:
    original_image_tensor = images_batch[0] # Prendiamo la singola immagine dal batch
    label_index = labels_batch[0].numpy()   # Prendiamo il label (indice intero)
    class_name = class_names[label_index]

    # La directory di output per la classe corrente
    current_class_output_dir = os.path.join(output_augmented_dir, class_name)

    # Nome base per l'immagine originale (può essere utile per tracciamento)
    # Se hai i nomi dei file originali, potresti usarli qui.
    # Altrimenti, usiamo un contatore.
    base_filename = f"original_{original_image_counter:05d}"
    original_image_counter += 1

    # Salviamo anche l'immagine originale (opzionale, ma utile per confronto)
    # Le immagini caricate da image_dataset_from_directory sono in formato float32 [0,255]
    # Per salvarle come immagini standard, le convertiamo in uint8
    # original_image_to_save = tf.cast(original_image_tensor, tf.uint8)
    # keras.utils.save_img(
    #     os.path.join(current_class_output_dir, f"{base_filename}_original.png"),
    #     original_image_to_save.numpy()
    # )

    # Applichiamo la data augmentation N volte per ogni immagine originale
    for i in range(num_augmentations_per_image):
        # Applica l'augmentation. 'training=True' è importante.
        # Aggiungiamo una dimensione batch perché i layer di augmentation si aspettano un batch.
        augmented_image_tensor = data_augmentation_layers(tf.expand_dims(original_image_tensor, 0), training=True)
        augmented_image_tensor = augmented_image_tensor[0] # Rimuoviamo la dimensione batch

        # Convertiamo l'immagine augmentata in uint8 per il salvataggio
        augmented_image_to_save = tf.cast(augmented_image_tensor, tf.uint8)

        # Definiamo il nome del file per l'immagine augmentata
        augmented_filename = f"{base_filename}_aug_{i+1:02d}.png"
        output_path = os.path.join(current_class_output_dir, augmented_filename)

        # Salviamo l'immagine augmentata
        keras.utils.save_img(output_path, augmented_image_to_save.numpy())

    if original_image_counter % 50 == 0: # Stampa un aggiornamento ogni 50 immagini originali
        print(f"Processate e augmentate {original_image_counter} immagini originali...")

print(f"\nProcesso di data augmentation e salvataggio completato.")
print(f"Le immagini augmentate sono state salvate in '{output_augmented_dir}'.")

# --- 6. Visualizzazione di Alcune Immagini Augmentate (Opzionale) ---
# Carichiamo qualche immagine salvata per verifica
if class_names: # Se ci sono classi
    first_class_name = class_names[0]
    augmented_sample_dir = os.path.join(output_augmented_dir, first_class_name)
    if os.path.exists(augmented_sample_dir):
        sample_augmented_images = [os.path.join(augmented_sample_dir, f) for f in os.listdir(augmented_sample_dir)[:min(9, len(os.listdir(augmented_sample_dir)))]]

        if sample_augmented_images:
            plt.figure(figsize=(10, 10))
            for i, img_path in enumerate(sample_augmented_images):
                img = keras.utils.load_img(img_path)
                ax = plt.subplot(3, 3, i + 1)
                plt.imshow(img)
                plt.title(f"Augmented {i+1}")
                plt.axis("off")
            plt.suptitle(f"Esempio di Immagini Augmentate Salvate per la Classe: {first_class_name}")
            plt.show()
        else:
            print(f"Nessuna immagine augmentata trovata in {augmented_sample_dir} per la visualizzazione.")
    else:
        print(f"La directory {augmented_sample_dir} non esiste per la visualizzazione.")
else:
    print("Nessuna classe trovata, impossibile visualizzare campioni.")
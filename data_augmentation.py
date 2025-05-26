import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt # Opzionale, per visualizzazione
import numpy as np
import os
from tqdm import tqdm # Per la barra di avanzamento
import shutil # Per creare/pulire directory

# --- 1. Definizione dei Parametri ---
dataset_dir = './dataset' 
output_augmented_dir = './augmented_dataset'

image_height = 180
image_width = 180
batch_size = 1 # Processo un'immagine alla volta per salvarla individualmente
seed = 42
num_augmentations_per_image = 5 # Quante versioni augmentate generare per ogni immagine originale

# --- 2. Preparazione Directory di Output ---
if os.path.exists(output_augmented_dir):
    print(f"La directory di output '{output_augmented_dir}' esiste già.")
    # ATTENZIONE: questo cancellerà il contenuto della directory!
    user_input = input(f"Vuoi cancellare e ricreare la directory '{output_augmented_dir}'? (s/N): ")
    if user_input.lower() == 's':
        shutil.rmtree(output_augmented_dir)
        print(f"Directory '{output_augmented_dir}' cancellata.")
    else:
        print("Operazione annullata. Modifica 'output_augmented_dir' o gestisci la directory esistente.")
        exit()

# Non è necessario ricreare la directory principale qui, lo farò per le sottoclassi.
# os.makedirs(output_augmented_dir, exist_ok=True)
# print(f"Directory di output '{output_augmented_dir}' pronta.")

# Sposto le immagini non di zanzare in una sottodirectory accorpando tutte le altre classi

folders = []
for entry in os.listdir(dataset_dir):
    # Verifichiamo se l'entry è una directory
    if os.path.isdir(os.path.join(dataset_dir, entry)):
        folders.append(entry)
folders.remove('Mosquito')  # Rimuovo la cartella delle zanzare dalla lista
print(f"Trovate {len(folders)} cartelle di immagini non di zanzare: {folders}")
# Creiamo una directory per le immagini non di zanzare
os.makedirs(os.path.join(dataset_dir, 'Not_Mosquito'), exist_ok=True)
non_zanzare_dir = os.path.join(dataset_dir, 'Not_Mosquito')
# Spostiamo le immagini non di zanzare in questa directory
for folder in folders:
    folder = os.path.join(dataset_dir, folder)
    nome_cartella = os.path.basename(folder)  # Ottengo il nome della cartella
    for nome_file in tqdm(os.listdir(folder), desc=f"Processando cartella '{nome_cartella}'"):
        percorso_file = os.path.join(folder, nome_file)
        if os.path.isfile(percorso_file):
            # Aggiungo il nome della cartella di partenza per evitare conflitti di nomi
            nuovo_nome_file = f"{nome_cartella}_{nome_file}"
            nuovo_percorso_file = os.path.join(non_zanzare_dir, nuovo_nome_file)
            counter = 1
            base, estensione = os.path.splitext(nuovo_percorso_file)
            while os.path.exists(nuovo_percorso_file):
                nuovo_nome = f"{base}_{counter}{estensione}"
                nuovo_percorso_file = os.path.join(non_zanzare_dir, nuovo_nome)
                counter += 1

            shutil.move(percorso_file, nuovo_percorso_file)
    # Rimuovo la cartella vuota dopo lo spostamento
    shutil.rmtree(folder)



# --- 3. Caricamento del Dataset Completo ---
# Carico l'intero dataset senza suddivisione in train/validation
# shuffle=False è importante per associare correttamente le immagini originali alle loro augmentations
# sebbene per il solo salvataggio potrebbe non essere strettamente necessario,
# è buona prassi se si volesse tracciarne l'origine.
full_dataset = keras.utils.image_dataset_from_directory(
    dataset_dir,
    labels='inferred',
    label_mode='int', # Uso 'int' per ottenere indici di classe diretti
    image_size=(image_height, image_width),
    interpolation='nearest',
    batch_size=batch_size, # Carico un'immagine alla volta
    shuffle=False, # Importante per mantenere l'ordine se necessario
    seed=seed
)

class_names = full_dataset.class_names
print("Nomi delle classi trovate:", class_names)

# Creo le directory di output per ogni classe
for class_name in class_names:
    class_output_dir = os.path.join(output_augmented_dir, class_name)
    os.makedirs(class_output_dir, exist_ok=True)
print(f"Directory di output per le classi create in '{output_augmented_dir}'.")


# --- 4. Data Augmentation Layers ---
# Definisco i layer di data augmentation
data_augmentation_layers = keras.Sequential(
    [
        layers.RandomFlip("horizontal", seed=seed),
        layers.RandomRotation(0.2, seed=seed), # Aumenta la rotazione
        layers.RandomZoom(0.2, seed=seed),   # Aumenta lo zoom
        layers.RandomContrast(0.2, seed=seed), # Aumenta il contrasto
        layers.RandomBrightness(0.2, seed=seed) # Assicurarsi che sia supportato dalla propria versione di TF
    ],
    name="data_augmentation",
)

# --- 5. Processo di Augmentation e Salvataggio ---
print("\nInizio processo di data augmentation e salvataggio...")

# Contatore per i nomi dei file originali per tracciamento
original_image_counter = 0

# Itero su ogni batch del dataset originale (ogni batch contiene 1 immagine in questo caso)
for images_batch, labels_batch in tqdm(full_dataset, desc="Processando immagini originali"):
    original_image_tensor = images_batch[0] # Prendo la singola immagine dal batch
    label_index = labels_batch[0].numpy()   # Prendo la label (indice intero)
    class_name = class_names[label_index]

    # La directory di output per la classe corrente
    current_class_output_dir = os.path.join(output_augmented_dir, class_name)

    # Nome base per l'immagine originale (può essere utile per tracciamento)
    base_filename = f"original_{original_image_counter:05d}"
    original_image_counter += 1

    # Salviamo anche l'immagine originale (opzionale, ma utile per confronto)
    # Le immagini caricate da image_dataset_from_directory sono in formato float32 [0,255]
    # Per salvarle come immagini standard, le convertiamo in uint8
    original_image_to_save = tf.cast(original_image_tensor, tf.uint8)
    keras.utils.save_img(
        os.path.join(current_class_output_dir, f"{base_filename}_original.png"),
        original_image_to_save.numpy()
    )

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

    #if original_image_counter % 50 == 0: # Stampa un aggiornamento ogni 50 immagini originali
    #    print(f"Processate e augmentate {original_image_counter} immagini originali...")

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
# convert_to_tflite.py

import tensorflow as tf
import os

# --- 1. PARAMETRI DI CONFIGURAZIONE ---

# Percorso del tuo modello Keras salvato
KERAS_MODEL_PATH = 'keras_training/keras_models_1607/CustomCNN_noDense_conv2D_final_model.keras' # MODIFICA QUESTO

# Percorso dove salvare il modello TFLite convertito
TFLITE_MODEL_PATH = 'tflite_models/tflite_models_1607/CustomCNN_noDense_conv2D_quant_int8.tflite' # MODIFICA QUESTO

# Percorso della cartella contenente i dati per la calibrazione
# (usa un piccolo subset del tuo set di addestramento)
CALIBRATION_DATA_PATH = 'augmented_dataset_splitted/train' # MODIFICA QUESTO

# Parametri delle immagini (devono corrispondere a quelli del modello)
IMG_SIZE = (96, 96)
BATCH_SIZE = 16
NUM_CALIBRATION_STEPS = 100 # (BATCH_SIZE * NUM_CALIBRATION_STEPS) = numero totale di immagini usate

# --- 2. CARICAMENTO DEL MODELLO KERAS ---

print(f"--- Caricamento del modello Keras da: {KERAS_MODEL_PATH} ---")
# Carica il modello Keras che include già il layer di Rescaling
model = tf.keras.models.load_model(KERAS_MODEL_PATH)
model.summary()


# --- 3. PREPARAZIONE DEL DATASET RAPPRESENTATIVO ---

print(f"--- Preparazione del dataset di calibrazione da: {CALIBRATION_DATA_PATH} ---")

# Questa funzione carica le immagini e le fornisce al convertitore.
# IMPORTANTE: NON viene applicata alcuna normalizzazione qui (es. /255.0),
# perché il modello stesso contiene già il layer di Rescaling.
def representative_dataset_gen():
  """Generatore per il dataset rappresentativo."""
  # Carica un batch di dati di calibrazione
  dataset = tf.keras.utils.image_dataset_from_directory(
      CALIBRATION_DATA_PATH,
      image_size=IMG_SIZE,
      batch_size=BATCH_SIZE,
      label_mode=None, # Non ci servono le etichette
      shuffle=True
  )
  
  # Itera per il numero di step di calibrazione
  for image_batch in dataset.take(NUM_CALIBRATION_STEPS):
    # Il convertitore si aspetta una lista o tupla
    print(f"Fornito un batch di calibrazione di shape: {image_batch.shape}")
    yield [image_batch]


# --- 4. CONFIGURAZIONE E ESECUZIONE DEL CONVERTITORE TFLITE ---

print("--- Inizio conversione a TFLite con quantizzazione int8 ---")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Abilita le ottimizzazioni standard (include la quantizzazione se fornito un representative_dataset)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Fornisci il dataset rappresentativo per la calibrazione int8
converter.representative_dataset = representative_dataset_gen

# Forza la quantizzazione int8 completa per la massima compatibilità
# (es. con microcontrollori o Edge TPU)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8  # Tipo di input grezzo (pixel 0-255)
converter.inference_output_type = tf.int8 # Tipo di output (logits quantizzati)

# Esegui la conversione
tflite_quant_model = converter.convert()

# --- 5. SALVATAGGIO DEL MODELLO TFLITE ---

try:
    with open(TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_quant_model)
    print("\n--- Conversione completata con successo! ---")
    print(f"Modello TFLite salvato in: {TFLITE_MODEL_PATH}")
    print(f"Dimensione del file: {os.path.getsize(TFLITE_MODEL_PATH) / 1024:.2f} KB")
except Exception as e:
    print(f"\nERRORE: Impossibile salvare il file del modello TFLite. {e}")
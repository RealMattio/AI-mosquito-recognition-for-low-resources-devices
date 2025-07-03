import tensorflow as tf
import numpy as np
import os
import argparse # Per gestire gli argomenti da riga di comando

<<<<<<< Updated upstream
# --- CONFIGURAZIONE ---
KERAS_MODELS_DIR = 'keras_models'
TFLITE_MODELS_DIR = 'tflite_models'
VALIDATION_DATA_DIR = 'augmented_dataset_splitted/validation'
IMG_SIZE = (224, 224)
NUM_CALIBRATION_SAMPLES = 100 # Numero di immagini da usare per la calibrazione
=======
# --- CONFIGURAZIONE (invariata) ---
KERAS_MODELS_DIR = 'keras_training/keras_models_0307'
TFLITE_MODELS_DIR = 'tflite_models/tflite_models_0307'
TRAINING_DATA_DIR = 'augmented_dataset_splitted/train' 
VALIDATION_DATA_DIR = 'augmented_dataset_splitted/validation'
IMG_SIZE = (96,96)
NUM_CALIBRATION_SAMPLES = 100
>>>>>>> Stashed changes

# Crea la cartella di output se non esiste
os.makedirs(TFLITE_MODELS_DIR, exist_ok=True)

# Mappa i nomi dei modelli alle loro funzioni di preprocessing specifiche
# Questo è fondamentale per una corretta calibrazione!
PREPROCESS_FUNCTIONS = {
    'MobileNetV2': tf.keras.applications.mobilenet_v2.preprocess_input,
    'ResNet50': tf.keras.applications.resnet50.preprocess_input,
    'NASNetMobile': tf.keras.applications.nasnet.preprocess_input
}

def convert_model(model_name):
    """
    Carica un modello Keras, lo converte in TFLite con quantizzazione INT8
    e lo salva su disco.
    """
    print(f"===== Inizio conversione per il modello: {model_name} =====")

    # 1. Trova il file del modello Keras
    # Cerca un file che inizi con il nome del modello nella cartella dei modelli
    keras_model_path = None
    for f in os.listdir(KERAS_MODELS_DIR):
        if f.startswith(model_name) and f.endswith('.keras'):
            keras_model_path = os.path.join(KERAS_MODELS_DIR, f)
            break
    
    if not keras_model_path:
        print(f"ERRORE: Nessun file .keras trovato per il modello '{model_name}' in '{KERAS_MODELS_DIR}'")
        return

    print(f"Trovato modello Keras: {keras_model_path}")

    # 2. Carica il modello Keras
    try:
        model = tf.keras.models.load_model(keras_model_path)
    except Exception as e:
        print(f"ERRORE: Impossibile caricare il modello Keras. Dettagli: {e}")
        return

    # 3. Prepara il dataset rappresentativo per la calibrazione
    print(f"Preparazione del dataset di calibrazione da '{VALIDATION_DATA_DIR}'...")
    
    try:
        validation_ds = tf.keras.utils.image_dataset_from_directory(
            VALIDATION_DATA_DIR,
            label_mode='int',
            seed=123,
            image_size=IMG_SIZE,
            batch_size=1 # La calibrazione richiede batch di dimensione 1
        ).take(NUM_CALIBRATION_SAMPLES)
    except Exception as e:
        print(f"ERRORE: Impossibile caricare i dati di validazione. Controlla il percorso. Dettagli: {e}")
        return

    # Ottieni la funzione di preprocessing corretta per questo modello
    preprocess_function = PREPROCESS_FUNCTIONS.get(model_name)
    if not preprocess_function:
        print(f"ERRORE: Funzione di preprocessing non definita per '{model_name}'")
        return

    def representative_data_gen():
        print("-> Esecuzione del generatore di dati rappresentativi...")
        for images, _ in validation_ds:
            # Applica il preprocessing specifico del modello
            # Il modello originale è stato addestrato con dati float, quindi la calibrazione
            # deve vedere dati nello stesso formato (es. da -1 a 1).
            preprocessed_images = preprocess_function(tf.cast(images, tf.float32))
            yield [preprocessed_images]

    # 4. Configura il convertitore TFLite
    print("Configurazione del convertitore TFLite con quantizzazione INT8...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    
    # Forza l'input e l'output del modello finale a essere INT8 per la massima
    # compatibilità con i microcontrollori
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    # 5. Esegui la conversione
    print("Avvio del processo di conversione (potrebbe richiedere qualche minuto)...")
    try:
        tflite_quant_model = converter.convert()
        print("Conversione completata con successo!")
    except Exception as e:
        print(f"ERRORE durante la conversione TFLite. Dettagli: {e}")
        return

    # 6. Salva il modello TFLite
    output_filename = f"{model_name}_quant_int8.tflite"
    output_path = os.path.join(TFLITE_MODELS_DIR, output_filename)
    
    with open(output_path, 'wb') as f:
        f.write(tflite_quant_model)
        
    print(f"Modello TFLite salvato in: {output_path}")
    print(f"Dimensione del file: {os.path.getsize(output_path) / 1024:.2f} KB")
    print("-" * 30)


if __name__ == '__main__':
    # Configura gli argomenti della riga di comando
    parser = argparse.ArgumentParser(description="Converti un modello Keras in formato TFLite con quantizzazione INT8.")
    parser.add_argument(
        '--model', 
        type=str, 
        required=True,
        choices=['MobileNetV2', 'ResNet50', 'NASNetMobile'],
        help="Il nome del modello da convertire."
    )
    
    args = parser.parse_args()
    run_conversion_process(args)

# Esegui il file con:
# python benchmark/keras_to_tflite.py --model MobileNetV2

# converter.py
# Data: 9 luglio 2025

import os
import tensorflow as tf
import numpy as np
import argparse

# --- CONFIGURAZIONE ---
# NOTA: Devi comunque specificare un percorso per i dati di validazione
# che servono per la calibrazione della quantizzazione.
VALIDATION_DATA_DIR = 'augmented_dataset_splitted/validation' 
IMG_SIZE = (96, 96)
NUM_CALIBRATION_SAMPLES = 50

# Dizionario con le funzioni di preprocessing per la calibrazione
PREPROCESS_FUNCTIONS = {
    'MobileNetV2': tf.keras.applications.mobilenet_v2.preprocess_input,
    'ResNet50': tf.keras.applications.resnet50.preprocess_input,
    'NASNetMobile': tf.keras.applications.nasnet.preprocess_input,
    'SeparableCustomCNN': lambda x: x / 255.0,
    'CustomCNN': lambda x: x / 255.0
}

def convert_single_model(source_dir, model_name, dest_dir):
    """
    Carica un singolo modello Keras, lo converte in TFLite INT8 e lo salva.
    """
    print(f"--- Inizio processo per il modello: {model_name} ---")

    # 1. Trova e carica il modello Keras
    keras_model_path = None
    try:
        for f in os.listdir(source_dir):
            if f.startswith(model_name) and f.endswith('.keras'):
                keras_model_path = os.path.join(source_dir, f)
                break
        if not keras_model_path:
            raise FileNotFoundError(f"Nessun file .keras che inizia con '{model_name}' trovato in '{source_dir}'")

        print(f"Trovato modello Keras: {keras_model_path}")
        print("Caricamento del modello...")
        model = tf.keras.models.load_model(keras_model_path)
        print("Modello caricato con successo!")

    except Exception as e:
        print(f"\nERRORE: Impossibile caricare il modello Keras.")
        print(f"Dettagli: {e}")
        #print("Questo è probabilmente un problema di incompatibilità di versione tra quando il modello è stato salvato e l'ambiente attuale.")
        return

    # 2. Prepara il dataset di calibrazione per la quantizzazione
    print(f"Preparazione di {NUM_CALIBRATION_SAMPLES} campioni di calibrazione da '{VALIDATION_DATA_DIR}'...")
    try:
        calibration_ds = tf.keras.utils.image_dataset_from_directory(
            VALIDATION_DATA_DIR,
            label_mode='int', seed=123, image_size=IMG_SIZE, batch_size=1
        ).take(NUM_CALIBRATION_SAMPLES)
    except FileNotFoundError:
        print(f"ERRORE: Cartella dei dati di validazione non trovata: '{VALIDATION_DATA_DIR}'.")
        return

    def representative_data_gen():
        # Assicurati che il nome del modello sia nel dizionario
        base_model_name = model_name.split('_')[0]
        if base_model_name not in PREPROCESS_FUNCTIONS:
             raise ValueError(f"Funzione di preprocessing non trovata per {base_model_name}")
        
        preprocess_function = PREPROCESS_FUNCTIONS[base_model_name]
        for images, _ in calibration_ds:
            yield [preprocess_function(tf.cast(images, tf.float32))]

    # 3. Configura il convertitore e converti in TFLite INT8
    print("Avvio conversione e quantizzazione INT8...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    try:
        tflite_quant_model = converter.convert()
        print("Conversione TFLite completata.")
    except Exception as e:
        print(f"ERRORE durante la conversione TFLite: {e}")
        return

    # 4. Salva il modello convertito
    os.makedirs(dest_dir, exist_ok=True)
    output_filename = f"{model_name}_quant_int8.tflite"
    output_path = os.path.join(dest_dir, output_filename)
    
    with open(output_path, 'wb') as f:
        f.write(tflite_quant_model)
        
    print(f"\nSUCCESSO! Modello TFLite salvato in: {output_path}")
    print(f"Dimensione file: {os.path.getsize(output_path) / 1024:.2f} KB")

if __name__ == '__main__':
    # Configura gli argomenti da riga di comando
    parser = argparse.ArgumentParser(description="Converti un singolo modello Keras in formato TFLite con quantizzazione INT8.")
    parser.add_argument('--source_dir', type=str, required=True, help="La cartella sorgente contenente il modello .keras.")
    parser.add_argument('--model_name', type=str, required=True, help="Il nome base del modello da cercare e convertire (es. 'ResNet50').")
    parser.add_argument('--dest_dir', type=str, required=True, help="La cartella di destinazione per il file .tflite salvato.")
    
    args = parser.parse_args()
    
    # Esegui la funzione di conversione
    convert_single_model(args.source_dir, args.model_name, args.dest_dir)

'''
# Esempio di utilizzo:
python benchmark/keras_to_tflite_simplified.py --source_dir keras_training/keras_models_0907 --model_name CustomCNN --dest_dir tflite_models/tflite_models_0907
'''
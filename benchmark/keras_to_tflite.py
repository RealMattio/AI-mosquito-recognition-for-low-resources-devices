# Data: 1 luglio 2025

import tensorflow as tf
import numpy as np
import os
import argparse
import tensorflow_model_optimization as tfmot

# --- CONFIGURAZIONE (invariata) ---
KERAS_MODELS_DIR = 'keras_training/keras_models_2506'
TFLITE_MODELS_DIR = 'tflite_models/tflite_models_2506'
TRAINING_DATA_DIR = 'augmented_dataset_splitted/train' 
VALIDATION_DATA_DIR = 'augmented_dataset_splitted/validation'
IMG_SIZE = (224, 224)
NUM_CALIBRATION_SAMPLES = 100

os.makedirs(TFLITE_MODELS_DIR, exist_ok=True)

PREPROCESS_FUNCTIONS = {
    'MobileNetV2': tf.keras.applications.mobilenet_v2.preprocess_input,
    'ResNet50': tf.keras.applications.resnet50.preprocess_input,
    'NASNetMobile': tf.keras.applications.nasnet.preprocess_input
}

# --- MODIFICA 1: L'architettura ora è "nuda", senza preprocessing ---
def build_prunable_model_architecture(model_name, num_classes=2):
    """
    Costruisce l'architettura del modello SENZA il layer di preprocessing.
    Questo modello accetta in input immagini con pixel [0, 255].
    """
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
    # L'input ora rappresenta direttamente l'immagine non normalizzata
    inputs = tf.keras.Input(shape=input_shape)
    
    if 'ResNet50' in model_name:
        base_model = tf.keras.applications.ResNet50(input_shape=input_shape, include_top=False, weights=None)
    elif 'MobileNetV2' in model_name:
        base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights=None)
    elif 'NASNetMobile' in model_name:
        base_model = tf.keras.applications.NASNetMobile(input_shape=input_shape, include_top=False, weights=None)
    else:
        raise ValueError(f"Architettura non definita per il modello: {model_name}")

    # Passiamo l'input direttamente al modello base
    x = base_model(inputs, training=False)
    
    # Il classificatore personalizzato rimane lo stesso
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs, outputs)


# --- MODIFICA 2: La funzione di fine-tuning ora gestisce il preprocessing ---
def apply_pruning_and_finetune(model, model_name, target_sparsity, epochs, patience, train_ds, val_ds):
    print(f"\n--- Inizio Pruning con sparsità target del {target_sparsity*100:.0f}% ---")

    # Ottieni la funzione di preprocessing corretta
    preprocess_function = PREPROCESS_FUNCTIONS[model_name]
    
    # Applica la funzione di preprocessing ai dataset "al volo"
    train_ds_preprocessed = train_ds.map(lambda x, y: (preprocess_function(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    val_ds_preprocessed = val_ds.map(lambda x, y: (preprocess_function(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    pruning_params = {'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(target_sparsity, 0, -1)}
    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)
    
    model_for_pruning.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=1, restore_best_weights=True)
    ]
    
    print(f"Fine-tuning del modello potato per un massimo di {epochs} epoche...")
    # Addestra il modello con i dati pre-processati
    model_for_pruning.fit(train_ds_preprocessed, epochs=epochs, callbacks=callbacks, validation_data=val_ds_preprocessed, verbose=1)
    
    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    print("--- Pruning e fine-tuning completati ---")
    
    return model_for_export

def run_conversion_process(args):
    model_name = args.model
    print(f"===== Inizio processo per il modello: {model_name} =====")

    keras_model_path = None
    for f in os.listdir(KERAS_MODELS_DIR):
        if f.startswith(model_name) and f.endswith('.keras'):
            keras_model_path = os.path.join(KERAS_MODELS_DIR, f)
            break
    if not keras_model_path:
        #...
        return
    print(f"Trovato file dei pesi Keras: {keras_model_path}")
    
    try:
        print("Costruzione dell'architettura del modello 'nuda'...")
        # Chiama la nuova funzione per costruire il modello senza preprocessing
        model = build_prunable_model_architecture(model_name)
        
        print("Caricamento dei pesi addestrati...")
        model.load_weights(keras_model_path)
        print("Modello e pesi caricati con successo!")
    except Exception as e:
        print(f"ERRORE durante la costruzione del modello o il caricamento dei pesi: {e}")
        return

    if args.pruning:
        print("Caricamento dati per il fine-tuning...")
        train_ds = tf.keras.utils.image_dataset_from_directory(TRAINING_DATA_DIR, label_mode='int', seed=123, image_size=IMG_SIZE, batch_size=32).prefetch(tf.data.AUTOTUNE)
        val_ds = tf.keras.utils.image_dataset_from_directory(VALIDATION_DATA_DIR, label_mode='int', seed=123, image_size=IMG_SIZE, batch_size=32).prefetch(tf.data.AUTOTUNE)
        
        # Passa il nome del modello alla funzione per sapere quale preprocessing usare
        model = apply_pruning_and_finetune(model, model_name, args.target_sparsity, args.pruning_epochs, args.early_stopping_patience, train_ds, val_ds)
    
    # La logica di quantizzazione ora opera sul modello "nudo" (potato o no)
    print(f"\nPreparazione del dataset di calibrazione...")
    calibration_ds = tf.keras.utils.image_dataset_from_directory(VALIDATION_DATA_DIR, label_mode='int', seed=123, image_size=IMG_SIZE, batch_size=1).take(NUM_CALIBRATION_SAMPLES)
    
    def representative_data_gen():
        preprocess_function = PREPROCESS_FUNCTIONS[model_name]
        for images, _ in calibration_ds:
            # Il generatore deve ancora fornire dati pre-processati,
            # perché il modello nudo li riceverà dopo la funzione di preprocessing
            yield [preprocess_function(tf.cast(images, tf.float32))]
            
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # ... (il resto della conversione è identico) ...
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    
    tflite_quant_model = converter.convert()
    
    pruned_suffix = "_pruned" if args.pruning else ""
    output_filename = f"{model_name}{pruned_suffix}_quant_int8.tflite"
    output_path = os.path.join(TFLITE_MODELS_DIR, output_filename)
    
    with open(output_path, 'wb') as f:
        f.write(tflite_quant_model)
    print(f"\nModello TFLite salvato in: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Converti un modello Keras in TFLite, con pruning opzionale e quantizzazione INT8.")
    parser.add_argument('--model', type=str, required=True, choices=['MobileNetV2', 'ResNet50', 'NASNetMobile'], help="Il nome del modello da convertire.")
    parser.add_argument('--pruning', action='store_true', help="Attiva il pruning e il fine-tuning.")
    parser.add_argument('--target_sparsity', type=float, default=0.5, help="Sparsità target per il pruning (default: 0.5).")
    parser.add_argument('--pruning_epochs', type=int, default=15, help="Epoche massime per il fine-tuning (default: 15).")
    parser.add_argument('--early_stopping_patience', type=int, default=3, help="Pazienza per l'Early Stopping (default: 3).")
    
    args = parser.parse_args()
    run_conversion_process(args)
# Data: 7 luglio 2025

import tensorflow as tf
import numpy as np
import os
import argparse
import tensorflow_model_optimization as tfmot

# --- CONFIGURAZIONE (invariata) ---
KERAS_MODELS_DIR = 'keras_training/keras_models_0807'
TFLITE_MODELS_DIR = 'tflite_models/tflite_models_0807'
TRAINING_DATA_DIR = 'augmented_dataset_splitted/train' 
VALIDATION_DATA_DIR = 'augmented_dataset_splitted/validation'
IMG_SIZE = (96, 96)
NUM_CALIBRATION_SAMPLES = 100

os.makedirs(TFLITE_MODELS_DIR, exist_ok=True)

# --- MODIFICA: Aggiunto il preprocessing per il CustomCNN ---
PREPROCESS_FUNCTIONS = {
    'MobileNetV2': tf.keras.applications.mobilenet_v2.preprocess_input,
    'ResNet50': tf.keras.applications.resnet50.preprocess_input,
    'NASNetMobile': tf.keras.applications.nasnet.preprocess_input,
    # Il preprocessing per il CustomCNN è una semplice normalizzazione a [0, 1]
    'CustomCNN': lambda x: x / 255.0
}

# --- MODIFICA: La funzione ora sa come costruire l'architettura CustomCNN ---
def build_prunable_model_architecture(model_name, num_classes=2):
    """
    Costruisce l'architettura del modello SENZA il layer di preprocessing.
    """
    input_shape = (IMG_SIZE[0], IMG_SIZE[1], 3)
    inputs = tf.keras.Input(shape=input_shape)
    
    if model_name == 'CustomCNN':
        # Definizione dell'architettura "from scratch"
        x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(inputs)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        # Il classificatore
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
        
        return tf.keras.Model(inputs, outputs)

    # La logica per i modelli pre-addestrati rimane la stessa
    if 'ResNet50' in model_name:
        base_model = tf.keras.applications.ResNet50(input_shape=input_shape, include_top=False, weights=None)
    elif 'MobileNetV2' in model_name:
        base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape, include_top=False, weights=None)
    elif 'NASNetMobile' in model_name:
        base_model = tf.keras.applications.NASNetMobile(input_shape=input_shape, include_top=False, weights=None)
    else:
        raise ValueError(f"Architettura non definita per il modello: {model_name}")

    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    return tf.keras.Model(inputs, outputs)


# La funzione di fine-tuning non richiede modifiche, è già generica
def apply_pruning_and_finetune(model, model_name, target_sparsity, epochs, patience, train_ds, val_ds):
    print(f"\n--- Inizio Pruning con sparsità target del {target_sparsity*100:.0f}% ---")
    
    # Questa funzione ora troverà anche il preprocessing per 'CustomCNN'
    preprocess_function = PREPROCESS_FUNCTIONS[model_name]
    
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
    model_for_pruning.fit(train_ds_preprocessed, epochs=epochs, callbacks=callbacks, validation_data=val_ds_preprocessed, verbose=1)
    
    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    print("--- Pruning e fine-tuning completati ---")
    
    return model_for_export

def run_conversion_process(args):
    model_name = args.model
    print(f"===== Inizio processo per il modello: {model_name} =====")

    # La logica di caricamento pesi funziona per tutti i modelli, incluso il CustomCNN
    keras_model_path = None
    for f in os.listdir(KERAS_MODELS_DIR):
        if f.startswith(model_name) and f.endswith('.keras'):
            keras_model_path = os.path.join(KERAS_MODELS_DIR, f)
            break
    if not keras_model_path:
        print(f"ERRORE: Nessun file .keras trovato per il modello '{model_name}'")
        return
    print(f"Trovato file dei pesi Keras: {keras_model_path}")
    
    try:
        print("Costruzione dell'architettura del modello...")
        model = build_prunable_model_architecture(model_name)
        
        print("Caricamento dei pesi addestrati...")
        model.load_weights(keras_model_path)
        print("Modello e pesi caricati con successo!")
    except Exception as e:
        print(f"ERRORE durante la costruzione del modello o il caricamento dei pesi: {e}")
        return

    if args.pruning:
        # ... (codice invariato) ...
        print("Caricamento dati per il fine-tuning...")
        train_ds = tf.keras.utils.image_dataset_from_directory(TRAINING_DATA_DIR, label_mode='int', seed=123, image_size=IMG_SIZE, batch_size=32).prefetch(tf.data.AUTOTUNE)
        val_ds = tf.keras.utils.image_dataset_from_directory(VALIDATION_DATA_DIR, label_mode='int', seed=123, image_size=IMG_SIZE, batch_size=32).prefetch(tf.data.AUTOTUNE)
        model = apply_pruning_and_finetune(model, model_name, args.target_sparsity, args.pruning_epochs, args.early_stopping_patience, train_ds, val_ds)

    # La logica di quantizzazione non richiede modifiche
    print(f"\nPreparazione del dataset di calibrazione...")
    calibration_ds = tf.keras.utils.image_dataset_from_directory(VALIDATION_DATA_DIR, label_mode='int', seed=123, image_size=IMG_SIZE, batch_size=1).take(NUM_CALIBRATION_SAMPLES)
    
    # Anche questo generatore ora funziona per il CustomCNN grazie al dizionario aggiornato
    def representative_data_gen():
        preprocess_function = PREPROCESS_FUNCTIONS[model_name]
        for images, _ in calibration_ds:
            yield [preprocess_function(tf.cast(images, tf.float32))]
            
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
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
    parser.add_argument(
        '--model', type=str, required=True,
        # --- MODIFICA: Aggiunto 'CustomCNN' alle scelte possibili ---
        choices=['MobileNetV2', 'ResNet50', 'NASNetMobile', 'CustomCNN'],
        help="Il nome del modello da convertire."
    )
    parser.add_argument(
        '--pruning', action='store_true',
        help="Attiva il pruning e il fine-tuning prima della quantizzazione."
    )
    parser.add_argument(
        '--target_sparsity', type=float, default=0.5,
        help="Sparsità target per il pruning (default: 0.5)."
    )
    parser.add_argument(
        '--pruning_epochs', type=int, default=15,
        help="Epoche massime per il fine-tuning (default: 15)."
    )
    parser.add_argument(
        '--early_stopping_patience', type=int, default=3,
        help="Pazienza per l'Early Stopping (default: 3)."
    )
    
    args = parser.parse_args()
    run_conversion_process(args)

# Esempio d'uso:
# python benchmark/keras_to_tflite.py --model CustomCNN --pruning --target_sparsity 0.5 --pruning_epochs 15 --early_stopping_patience 3
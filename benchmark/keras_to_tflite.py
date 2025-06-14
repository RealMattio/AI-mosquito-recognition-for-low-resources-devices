import tensorflow as tf
import numpy as np
import os

# Importa la funzione di pre-processing specifica per il modello che stai usando
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as model_preprocess_input
# Se stessi usando ResNet50: from tensorflow.keras.applications.resnet50 import preprocess_input
# Se stessi usando NASNet: from tensorflow.keras.applications.nasnet import preprocess_input

# --- IMPOSTAZIONI DA PERSONALIZZARE ---
# Percorso al tuo modello .keras GIA' FINETUNATO
MODEL_NAME = 'ResNet50_0.9822'
MODEL_DIR = 'saved_models'
QUANTIZED_MODEL_DIR = 'tflite_models'
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME+ '.keras')
TRAIN_DIR = 'augmented_dataset_splitted/train'
# Usa le dimensioni di input standard per MobileNetV2, o quelle che hai usato nel fine-tuning
IMG_HEIGHT = 224
IMG_WIDTH = 224
COLOR_MODE = 'rgb' # I modelli su ImageNet sono a colori
# --- FINE IMPOSTAZIONI ---

# 1. Carica il tuo modello Keras fine-tunato
model = tf.keras.models.load_model(MODEL_PATH)
print("Modello .keras caricato correttamente.")

# 2. Funzione di pre-processing che usa la funzione specifica del modello
#    Questa è la modifica CRUCIALE!
def preprocess_image(image, label):
    image = tf.cast(image, tf.float32)
    # Applica la funzione di pre-processing di MobileNetV2 (scala i pixel a [-1, 1])
    image = model_preprocess_input(image)
    return image, label

# 3. Carica il dataset di training per la quantizzazione
train_dataset = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels='inferred',
    label_mode='int',
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    color_mode=COLOR_MODE,
    batch_size=1
)

# Applica la funzione di pre-processing corretta
train_dataset = train_dataset.map(preprocess_image)

# 4. Funzione generatore per il convertitore
def representative_dataset_gen():
    for images, _ in train_dataset.take(200):
        yield [images]

# 5. Converti e quantizza
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()

# 6. Salva il modello
output_filename = os.path.join(QUANTIZED_MODEL_DIR, MODEL_NAME + '_quantized.tflite')
with open(output_filename, 'wb') as f:
    f.write(tflite_quant_model)

print(f"\nModello convertito e salvato in '{output_filename}'")
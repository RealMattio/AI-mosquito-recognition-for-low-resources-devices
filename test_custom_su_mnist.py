import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

# --- 1. CARICAMENTO E PREPARAZIONE DEL DATASET MNIST ---

print("Caricamento e preparazione del dataset MNIST...")

# Carica il dataset completo
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Filtra per mantenere solo le cifre 0 e 1
train_mask = np.isin(y_train, [0, 1])
test_mask = np.isin(y_test, [0, 1])

x_train, y_train = x_train[train_mask], y_train[train_mask]
x_test, y_test = x_test[test_mask], y_test[test_mask]

# Preprocessing delle immagini:
# - Aggiungi una dimensione per il canale (da 60000x28x28 a 60000x28x28x1)
# - Normalizza i pixel a valori float32
x_train = np.expand_dims(x_train, axis=-1).astype(np.float32)
x_test = np.expand_dims(x_test, axis=-1).astype(np.float32)

print(f"Dataset pronto: {x_train.shape[0]} campioni di training, {x_test.shape[0]} campioni di test.")
print("-" * 50)


# --- 2. COSTRUZIONE DEL MODELLO CNN LEGGERO ---

print("Costruzione del modello CNN custom...")

input_shape = x_train.shape[1:] # (28, 28, 1)
num_classes = 2 # Classi 0 e 1

# model = models.Sequential([
#     # Il primo layer definisce la forma dell'input e normalizza i pixel a [0, 1]
#     layers.Input(shape=input_shape),
#     layers.Rescaling(1./255),
    
#     # Blocco Convoluzionale 1
#     layers.Conv2D(16, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
    
#     # Blocco Convoluzionale 2
#     layers.Conv2D(32, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
    
#     # Blocco Convoluzionale 3
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
    
#     # Classificatore
#     layers.Reshape((6400,1)),
#     layers.Conv1D(filters=64, kernel_size=6400, activation='relu', padding='valid'),
#     layers.Reshape((64,1)),
#     layers.Conv1D(2, 64, activation='softmax', padding='valid') 
# ], name="CustomCNN_MNIST")

def build_modular_cnn(input_shape=input_shape, num_classes=num_classes):
    """
    Costruisce una CNN modulare che si adatta a diverse dimensioni di input.
    """
    # 1. Definisci l'input del modello
    inputs = layers.Input(shape=input_shape)

    # 2. Costruisci la base convoluzionale
    x = layers.Rescaling(1./255)(inputs)
    x = layers.Conv2D(8, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    # L'output di questa base avrà una forma (H, W, C) che dipende dall'input
    conv_base_output = layers.Conv2D(32, (3, 3), activation='relu')(x)

    # 3. Calcola dinamicamente la dimensione per il flatten
    # Otteniamo la forma dell'output della base (es. (None, 10, 10, 64))
    shape = conv_base_output.shape
    # Calcoliamo la dimensione del vettore appiattito (es. 10 * 10 * 64 = 6400)
    flattened_size = shape[1] * shape[2] * shape[3]
    
    # 4. Costruisci il classificatore dinamico
    # Il Reshape ora usa la dimensione calcolata al volo
    x = layers.Reshape((flattened_size, 1))(conv_base_output)

    # Il kernel della Conv1D ora usa la dimensione calcolata al volo
    x = layers.Conv1D(filters=32, kernel_size=flattened_size, activation='relu', padding='valid')(x)

    # Il resto del classificatore
    x = layers.Reshape((32, 1))(x) # Questo 64 è fisso perché è il n° di filtri del layer precedente
    x = layers.Conv1D(num_classes, 32, activation='softmax', padding='valid')(x)
    
    # Flatten finale per ottenere l'output nella forma corretta (batch, num_classes)
    outputs = layers.Flatten()(x)

    # 5. Crea e restituisci il modello finale
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

# model = models.Sequential([
#     layers.Input(shape=input_shape),
#     layers.Rescaling(1./255), # Normalizza i pixel a [0, 1]

#     layers.SeparableConv2D(8, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
    
#     layers.SeparableConv2D(16, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
    
#     layers.GlobalAveragePooling2D(),
    
#     layers.Dense(num_classes, activation='softmax')
# ], name="CustomCNN_MNIST")
model = build_modular_cnn()

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
print("-" * 50)


# --- 3. ADDESTRAMENTO DEL MODELLO KERAS ---

print("Inizio addestramento del modello Keras...")
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test), batch_size=64)
print("-" * 50)


# --- 4. VALUTAZIONE DEL MODELLO KERAS ORIGINALE ---

print("Valutazione del modello Keras originale (FP32)...")
loss_fp32, accuracy_fp32 = model.evaluate(x_test, y_test, verbose=0)
print(f"-> Accuratezza (FP32): {accuracy_fp32 * 100:.2f}%")
print(f"-> Loss (FP32): {loss_fp32:.4f}")
print("-" * 50)
model.save('mnist_custom_cnn.keras')  # Salva il modello Keras per uso futuro

# --- 5. CONVERSIONE IN TFLITE CON QUANTIZZAZIONE INT8 ---

print("Conversione del modello in TFLite con quantizzazione INT8...")

# Crea un dataset rappresentativo per la calibrazione
def representative_data_gen():
    for image in tf.data.Dataset.from_tensor_slices(x_train).batch(1).take(100):
        yield [image]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_quant_model = converter.convert()

# Salva il modello TFLite su disco
tflite_model_path = "mnist_custom_cnn_quant.tflite"
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_quant_model)

print(f"Modello TFLite quantizzato salvato come '{tflite_model_path}'")
print(f"Dimensioni del file: {os.path.getsize(tflite_model_path) / 1024:.2f} KB")
print("-" * 50)


# --- 6. VALUTAZIONE DEL MODELLO TFLITE QUANTIZZATO ---

print("Valutazione del modello TFLite quantizzato (INT8)...")

interpreter = tf.lite.Interpreter(model_content=tflite_quant_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

correct_predictions = 0
from tqdm import tqdm
for i in tqdm(range(len(x_test))):
    # Ottieni l'immagine di test
    test_image = x_test[i]
    
    # Prepara l'input per il modello INT8
    # (Quantizza l'immagine nello stesso modo in cui il modello si aspetta)
    input_scale, input_zero_point = input_details["quantization"]
    test_image_quant = (test_image / input_scale) + input_zero_point
    test_image_quant = np.expand_dims(test_image_quant, axis=0).astype(input_details["dtype"])

    # Esegui l'inferenza
    interpreter.set_tensor(input_details['index'], test_image_quant)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details['index'])
    
    # Ottieni la predizione
    predicted_label = np.argmax(output_data)
    
    if predicted_label == y_test[i]:
        correct_predictions += 1

accuracy_int8 = correct_predictions / len(x_test)

print(f"-> Accuratezza (INT8): {accuracy_int8 * 100:.2f}%")
print("-" * 50)


# --- RIEPILOGO FINALE ---
print("\n===== RIEPILOGO METRICHE =====")
print(f"Accuratezza Modello Keras (FP32): {accuracy_fp32 * 100:.2f}%")
print(f"Accuratezza Modello TFLite (INT8): {accuracy_int8 * 100:.2f}%")
print("==============================")
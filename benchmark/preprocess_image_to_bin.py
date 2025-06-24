# Esempio di preprocess_image.py:
import tensorflow as tf
from PIL import Image
import numpy as np

IMG_PATH = 'augmented_dataset_splitted/test/Mosquito/original_00000_aug_02.png'
IMG_SIZE = 224
SAVING_NAME = 'benchmark/benchmark_image.bin'

img = Image.open(IMG_PATH).resize((IMG_SIZE, IMG_SIZE))
img_array = np.array(img, dtype=np.uint8) # Dati da 0 a 255

# Per un modello quantizzato INT8, il range tipico è -128 a 127
# La conversione standard è: (valore_uint8 - 128)
img_array_int8 = (img_array.astype(np.int32) - 128).astype(np.int8)

# Salva l'array binario grezzo
with open(SAVING_NAME, 'wb') as f:
    f.write(img_array_int8.tobytes())

print("Immagine pre-processata e salvata come image_data.bin")
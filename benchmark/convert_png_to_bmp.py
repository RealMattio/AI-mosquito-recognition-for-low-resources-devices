from PIL import Image

# --- CONFIGURAZIONE ---
INPUT_IMAGE_PATH = 'benchmark/original_00000_original.png'
IMG_SIZE = (224, 224) # Deve corrispondere alle dimensioni attese dal tuo modello
OUTPUT_BMP_PATH = 'benchmark/test_image_224.bmp' # Nome del file che caricheremo sulla scheda
# --------------------

print(f"Conversione di '{INPUT_IMAGE_PATH}' in formato BMP...")

# Apri l'immagine, convertila in RGB e ridimensionala
img = Image.open(INPUT_IMAGE_PATH).convert('RGB').resize(IMG_SIZE)

# Salva l'immagine in formato BMP
img.save(OUTPUT_BMP_PATH, 'bmp')

print(f"Immagine salvata con successo come '{OUTPUT_BMP_PATH}'")
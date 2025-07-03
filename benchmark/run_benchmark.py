import tensorflow as tf
import numpy as np
import os
import time
import json
from PIL import Image

import platform
import psutil
import cpuinfo
# pynvml è opzionale e verrà gestito con un try-except
try:
    import pynvml
except ImportError:
    pynvml = None

# --- CONFIGURAZIONE ---
KERAS_MODELS_DIR = 'keras_training/keras_models_2506'
TFLITE_MODELS_DIR = 'tflite_models/tflite_models_2506'
IMAGE_PATH = 'benchmark/original_00000_original.png' # Percorso dell'immagine di TEST copiata nella cartella benchmark
RESULTS_JSON_PATH = 'benchmark/benchmark_results_Raspberry_Pi_5.json'

NUM_INFERENCE_RUNS = 50
IMG_SIZE = (224, 224)

# Forza l'uso di Keras 3 perché i modelli sono addestrati con Keras 3.
# Questa linea è FONDAMENTALE per aumentare la probabilità che tf.keras.models.load_model
# funzioni correttamente in ambienti dove gli addestramenti sono stati forzati con Keras 2.
# os.environ['TF_USE_LEGACY_KERAS'] = '0'

def get_system_info():
    """Raccoglie e restituisce un dizionario con le informazioni hardware e software del sistema."""
    print("\n--- Raccolta Informazioni di Sistema ---")
    info = {}
    
    # Sistema Operativo
    info['os'] = {
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version()
    }
    
    # Processore (CPU)
    cpu_info = cpuinfo.get_cpu_info()
    info['cpu'] = {
        'brand': cpu_info.get('brand_raw', 'N/A'),
        'arch': cpu_info.get('arch_string_raw', 'N/A'),
        'cores_physical': psutil.cpu_count(logical=False),
        'cores_logical': psutil.cpu_count(logical=True)
    }
    
    # Memoria (RAM)
    ram_info = psutil.virtual_memory()
    info['ram'] = {
        'total_gb': round(ram_info.total / (1024**3), 2)
    }
    
    # Scheda Grafica (GPU)
    info['gpu'] = []
    # Prima, vediamo cosa rileva TensorFlow
    tf_gpus = tf.config.list_physical_devices('GPU')
    if tf_gpus:
        info['gpu'].append({'detected_by_tensorflow': [gpu.name for gpu in tf_gpus]})
    else:
        info['gpu'].append({'detected_by_tensorflow': 'None'})

    # Poi, proviamo a ottenere dettagli specifici per NVIDIA con pynvml
    if pynvml:
        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            nvidia_gpus = []
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                gpu_name = pynvml.nvmlDeviceGetName(handle)
                driver_version = pynvml.nvmlSystemGetDriverVersion()
                total_memory_gb = round(pynvml.nvmlDeviceGetMemoryInfo(handle).total / (1024**3), 2)
                nvidia_gpus.append({
                    'name': gpu_name,
                    'driver_version': driver_version,
                    'total_memory_gb': total_memory_gb
                })
            info['gpu'].append({'nvidia_details': nvidia_gpus})
            pynvml.nvmlShutdown()
        except Exception as e:
            info['gpu'].append({'nvidia_details': f'Error querying NVML: {e}'})
    else:
        info['gpu'].append({'nvidia_details': 'pynvml library not found'})
        
    print("Informazioni di sistema raccolte.")
    return info


def preprocess_image(image_path, size):
    """Carica e pre-processa una singola immagine."""
    # Nota: non applichiamo la normalizzazione qui, perché il modello Keras
    # salvato ha già il layer di preprocessing al suo interno.
    img = Image.open(image_path).convert('RGB').resize(size)
    img_array = np.array(img, dtype=np.float32)
    # Aggiungi la dimensione del batch -> (1, 224, 224, 3)
    return np.expand_dims(img_array, axis=0)


def benchmark_keras_model(model_path, image_data):
    """Esegue il benchmark su un modello Keras caricato direttamente."""
    print(f"\n--- Benchmark Keras: {os.path.basename(model_path)} ---")
    try:
        # Utilizziamo il caricamento diretto standard di Keras
        print(f"Tentativo di caricare il modello con tf.keras.models.load_model...")
        model = tf.keras.models.load_model(model_path)
        print("Modello Keras caricato con successo.")
    except Exception as e:
        print(f"ERRORE durante il caricamento del modello Keras: {e}")
        print("Questo errore è probabilmente dovuto a un'incompatibilità di versione tra salvataggio e caricamento.")
        return None

    # "Riscaldamento" del modello (prima inferenza non misurata)
    _ = model.predict(image_data, verbose=0)
    
    times = []
    print(f"Esecuzione di {NUM_INFERENCE_RUNS} inferenze...")
    for _ in range(NUM_INFERENCE_RUNS):
        start_time = time.perf_counter()
        _ = model.predict(image_data, verbose=0)
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000) # Salvato in millisecondi

    # Calcolo delle statistiche
    total_time = np.sum(times)
    avg_time = np.mean(times)
    std_dev = np.std(times)
    
    print(f"Completato. Tempo medio: {avg_time:.2f} ms")
    
    return {
        'total_time_ms': total_time,
        'average_time_ms': avg_time,
        'std_dev_ms': std_dev,
        'individual_times_ms': times
    }

def benchmark_tflite_model(model_path, image_data):
    """Esegue il benchmark su un modello TFLite (invariato)."""
    print(f"\n--- Benchmark TFLite: {os.path.basename(model_path)} ---")
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print("Modello TFLite caricato con successo.")
    except Exception as e:
        print(f"ERRORE durante il caricamento del modello TFLite: {e}")
        return None

    input_details = interpreter.get_input_details()[0]
    
    # Prepara l'immagine per il modello quantizzato (INT8)
    if input_details['dtype'] == np.int8:
        input_scale, input_zero_point = input_details['quantization']
        # La formula corretta per la quantizzazione
        input_data = (image_data / input_scale + input_zero_point).astype(np.int8)
    else: # Se il modello fosse float
        input_data = image_data.astype(np.float32)

    interpreter.set_tensor(input_details['index'], input_data)
    
    # "Riscaldamento"
    interpreter.invoke()
    
    times = []
    print(f"Esecuzione di {NUM_INFERENCE_RUNS} inferenze...")
    for _ in range(NUM_INFERENCE_RUNS):
        start_time = time.perf_counter()
        interpreter.invoke()
        end_time = time.perf_counter()
        times.append((end_time - start_time) * 1000)

    total_time = np.sum(times)
    avg_time = np.mean(times)
    std_dev = np.std(times)
    
    print(f"Completato. Tempo medio: {avg_time:.2f} ms")

    return {
        'total_time_ms': total_time,
        'average_time_ms': avg_time,
        'std_dev_ms': std_dev,
        'individual_times_ms': times
    }

if __name__ == '__main__':
    # 1. Raccogli le info di sistema all'inizio
    system_info = get_system_info()

    # Stampa un riepilogo a schermo
    print(f"  OS: {system_info['os']['system']} {system_info['os']['release']}")
    print(f"  CPU: {system_info['cpu']['brand']}")
    print(f"  RAM: {system_info['ram']['total_gb']} GB")
    print("-" * 30)
    
    if not os.path.exists(IMAGE_PATH):
        raise FileNotFoundError(f"Immagine di test non trovata in: {IMAGE_PATH}.")

    image_data_float = preprocess_image(IMAGE_PATH, IMG_SIZE)
    
    benchmark_runs = {} # Dizionario per i soli risultati dei modelli

    # --- Benchmark Modelli Keras ---
    if os.path.exists(KERAS_MODELS_DIR):
        for filename in sorted(os.listdir(KERAS_MODELS_DIR)):
            if filename.endswith('.keras'):
                model_path = os.path.join(KERAS_MODELS_DIR, filename)
                results = benchmark_keras_model(model_path, image_data_float)
                if results:
                    benchmark_runs[f"{filename}_Keras"] = results
    else:
        print(f"ATTENZIONE: Cartella dei modelli Keras non trovata: {KERAS_MODELS_DIR}")

    # --- Benchmark Modelli TFLite ---
    if os.path.exists(TFLITE_MODELS_DIR):
        for filename in sorted(os.listdir(TFLITE_MODELS_DIR)):
            if filename.endswith('.tflite'):
                model_path = os.path.join(TFLITE_MODELS_DIR, filename)
                results = benchmark_tflite_model(model_path, image_data_float)
                if results:
                    benchmark_runs[f"{filename}_TFLite"] = results
    else:
         print(f"ATTENZIONE: Cartella dei modelli TFLite non trovata: {TFLITE_MODELS_DIR}")

    # 2. Crea il dizionario finale completo
    final_results_json = {
        'system_info': system_info,
        'benchmark_runs': benchmark_runs
    }

    # 3. Salva il nuovo JSON strutturato
    if benchmark_runs:
        print(f"\nSalvataggio dei risultati completi del benchmark in: {RESULTS_JSON_PATH}")
        with open(RESULTS_JSON_PATH, 'w') as f:
            # Usiamo un custom encoder per gestire i tipi numpy se necessario
            class NpEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super(NpEncoder, self).default(obj)
            
            json.dump(final_results_json, f, indent=4, cls=NpEncoder)
        print("Salvataggio completato.")
    else:
        print("\nNessun benchmark eseguito. Nessun risultato da salvare.")
# Script di benchmark DEFINITIVO, con i percorsi corretti per il filesystem del Portenta H7

import time
import ml
import image
from ulab import numpy as np
import gc

# --- CONFIGURAZIONE ---
# Abbiamo aggiunto il percorso assoluto '/flash/' per essere espliciti.
MODEL_PATH = "/flash/MobileNetV2_quant_int8.tflite"
IMAGE_PATH = "/flash/test_image_224.bmp"
NUM_INFERENCE_RUNS = 50
# --------------------


def run_inference_benchmark(model_path, image_path):
    """
    Esegue il benchmark di inferenza usando i percorsi corretti.
    """
    print(f"\n--- Benchmark TFLite: {model_path} ---")

    # 1. Forziamo la pulizia della memoria PRIMA di caricare il modello
    #    È una buona pratica che teniamo per massimizzare la memoria libera.
    print("-> Esecuzione del Garbage Collector...")
    gc.collect()

    # 2. Caricamento del Modello
    try:
        model = ml.Model(model_path, load_to_fb=True)
        print("-> Modello caricato con successo!")
    except Exception as e:
        print(f"ERRORE: Impossibile creare l'oggetto ml.Model.\nL'errore è: {e}")
        return None

    # 3. Caricamento dell'Immagine
    try:
        img = image.Image(image_path, copy_to_fb=True)
        print("-> Immagine di test caricata con successo.")
    except Exception as e:
        print(f"ERRORE: Impossibile caricare l'immagine.\nL'errore è: {e}")
        return None

    # 4. Esecuzione del Benchmark
    try:
        print("-> Esecuzione di riscaldamento (warm-up)...")
        _ = model.classify(img)

        print(f"-> Esecuzione di {NUM_INFERENCE_RUNS} inferenze per il benchmark...")
        times_array = np.zeros(NUM_INFERENCE_RUNS, dtype=np.float)

        for i in range(NUM_INFERENCE_RUNS):
            start_time = time.ticks_us()
            model.classify(img)
            end_time = time.ticks_us()
            times_array[i] = time.ticks_diff(end_time, start_time) / 1000.0

        print("-> Benchmark completato.")

        avg_time = np.mean(times_array)
        std_dev = np.std(times_array)
        total_time = np.sum(times_array)

        return {
            'total_time_ms': total_time,
            'average_time_ms': avg_time,
            'std_dev_ms': std_dev
        }
    except Exception as e:
        print(f"ERRORE durante l'esecuzione di model.classify(): \n{e}")
        return None


# --- ESECUZIONE PRINCIPALE ---
print("Avvio del benchmark di inferenza sul Portenta H7...")
benchmark_metrics = run_inference_benchmark(MODEL_PATH, IMAGE_PATH)
if benchmark_metrics:
    print("\n--- RISULTATI DEL BENCHMARK ---")
    print(f"Tempo medio di inferenza: {benchmark_metrics['average_time_ms']:.2f} ms")
    print(f"Deviazione Standard:       {benchmark_metrics['std_dev_ms']:.2f} ms")
    print(f"Tempo totale ({NUM_INFERENCE_RUNS} esecuzioni): {benchmark_metrics['total_time_ms']:.0f} ms")
    print("---------------------------------")
else:
    print("\nBenchmark non completato a causa di un errore precedente.")

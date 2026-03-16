import os
import tensorflow as tf
from tensorflow import keras

def pulisci_cartella_modelli(cartella_input, cartella_output):
    """
    Legge tutti i modelli .keras in una cartella, risolve le dipendenze
    di preprocess_input e li salva in versione alleggerita.
    """
    
    # 1. Creiamo la cartella di output se non esiste
    if not os.path.exists(cartella_output):
        os.makedirs(cartella_output)
        print(f"Creata cartella di output: {cartella_output}")

    # 2. Mappiamo i nomi dei file alle funzioni corrette
    # Questo dizionario dice allo script: "Se il nome del file contiene X, 
    # usa la funzione preprocess_input del modulo Y"
    mappa_preprocessing = {
        "MobileNetV2": tf.keras.applications.mobilenet_v2.preprocess_input,
        "MobileNet_": tf.keras.applications.mobilenet.preprocess_input, # L'underscore evita conflitti con V2
        "NASNetMobile": tf.keras.applications.nasnet.preprocess_input,
        "ResNet50": tf.keras.applications.resnet50.preprocess_input
    }

    # 3. Cerchiamo tutti i file .keras nella cartella
    file_modelli = [f for f in os.listdir(cartella_input) if f.endswith('.keras')]
    
    if not file_modelli:
        print(f"Nessun file .keras trovato in {cartella_input}")
        return

    print(f"Trovati {len(file_modelli)} modelli da alleggerire.\n")

    # 4. Cicliamo su ogni modello
    for nome_file in file_modelli:
        percorso_input_completo = os.path.join(cartella_input, nome_file)
        percorso_output_completo = os.path.join(cartella_output, nome_file)
        
        print(f"Elaborazione di: {nome_file}")
        
        # Identifichiamo quale funzione usare per questo specifico file
        funzione_corretta = None
        for chiave, funzione in mappa_preprocessing.items():
            if chiave in nome_file:
                funzione_corretta = funzione
                break # Trovata la corrispondenza, usciamo dal mini-ciclo
        
        # Prepara gli oggetti personalizzati. Se non è uno dei modelli pre-addestrati 
        # (es. le tue CustomCNN), passeremo un dizionario vuoto.
        oggetti_personalizzati = {}
        if funzione_corretta:
            oggetti_personalizzati = {'preprocess_input': funzione_corretta}
            print(f"  -> Usato preprocess_input per: {chiave}")
        else:
             print("  -> Nessun preprocessing specifico richiesto.")

        try:
            # Carichiamo il modello passando la funzione corretta!
            modello = keras.models.load_model(
                percorso_input_completo, 
                compile=False,
                custom_objects=oggetti_personalizzati
            )
            
            # Salviamo senza ottimizzatore
            modello.save(percorso_output_completo, include_optimizer=False)
            
            # Calcoliamo il risparmio di spazio (opzionale, ma carino da vedere!)
            peso_old = os.path.getsize(percorso_input_completo) / (1024 * 1024)
            peso_new = os.path.getsize(percorso_output_completo) / (1024 * 1024)
            print(f"  -> Fatto! Ridotto da {peso_old:.1f} MB a {peso_new:.1f} MB.\n")
            
        except Exception as e:
             print(f"  -> ERRORE con {nome_file}: {e}\n")

# ==========================================
# Istruzioni di implementazione:
# ==========================================

# Specifica le cartelle (modifica i percorsi se necessario)
cartella_modelli_pesanti = "./final_keras_models/full_training"
cartella_modelli_leggeri = "./final_keras_models/LightModels" # Ti consiglio di usare una cartella separata

# Avvia il processo!
pulisci_cartella_modelli(cartella_modelli_pesanti, cartella_modelli_leggeri)
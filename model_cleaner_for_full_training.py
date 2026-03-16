import keras
import os

def riduci_dimensione_modello(percorso_input, percorso_output):
    """
    Carica un modello Keras e lo salva nuovamente senza lo stato dell'ottimizzatore
    per ridurne le dimensioni sul disco.
    """
    # Controlliamo che il file di input esista per evitare errori
    if not os.path.exists(percorso_input):
        print(f"Errore: Il file {percorso_input} non è stato trovato.")
        return

    print(f"Caricamento del modello pesante da:\n-> {percorso_input}")
    
    # Carichiamo il modello. Usiamo compile=False perché per il semplice 
    # salvataggio non ci serve che il modello sia compilato per l'addestramento.
    modello = keras.models.load_model(percorso_input, compile=False)
    
    print(f"Salvataggio del modello alleggerito in:\n-> {percorso_output}")
    
    # Questa è la magia: salviamo il modello escludendo l'ottimizzatore!
    modello.save(percorso_output, include_optimizer=False)
    
    print("Operazione completata con successo! Il tuo modello ora è a dieta. 🎉\n")

# ==========================================
# Istruzioni di implementazione:
# Inserisci qui sotto i percorsi dei tuoi file
# ==========================================

modello="ResNet50"  # Sostituisci con il nome del tuo modello

# 1. Scrivi il percorso esatto del modello che vuoi alleggerire
percorso_modello_pesante = f"./final_keras_models/full_training/{modello}_ClassicDenseClassifier_final_model.keras"

# 2. Scrivi il nome e il percorso dove vuoi salvare la nuova versione
percorso_modello_leggero = f"./final_keras_models/LightModels/{modello}_full_weights.keras"

# Eseguiamo la funzione
riduci_dimensione_modello(percorso_modello_pesante, percorso_modello_leggero)
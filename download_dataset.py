import os
import shutil
import kagglehub


def copia_contenuto_cartella(sorgente, destinazione):
    """
    Copia tutto il contenuto (file e sottocartelle) da una cartella sorgente
    a una cartella di destinazione.

    :param sorgente: Percorso della cartella da cui copiare.
    :param destinazione: Percorso della cartella in cui copiare.
    """
    # 1. Controlla se la cartella sorgente esiste
    if not os.path.exists(sorgente):
        print(f"Errore: La cartella sorgente '{sorgente}' non esiste.")
        return

    # 2. Crea la cartella di destinazione se non esiste
    #    os.makedirs la crea ricorsivamente e non dà errore se esiste già (exist_ok=True)
    os.makedirs(destinazione, exist_ok=True)
    print(f"La cartella di destinazione '{destinazione}' è pronta.")

    # 3. Itera su tutti i file e le cartelle nella sorgente
    for nome_elemento in os.listdir(sorgente):
        percorso_sorgente_elemento = os.path.join(sorgente, nome_elemento)
        percorso_destinazione_elemento = os.path.join(destinazione, nome_elemento)

        try:
            # Se è una cartella, usa shutil.copytree per la copia ricorsiva
            if os.path.isdir(percorso_sorgente_elemento):
                shutil.copytree(percorso_sorgente_elemento, percorso_destinazione_elemento)
                print(f"Copiata cartella: '{nome_elemento}'")
            # Se è un file, usa shutil.copy2 per copiare il file e i metadati
            else:
                shutil.copy2(percorso_sorgente_elemento, percorso_destinazione_elemento)
                print(f"Copiato file: '{nome_elemento}'")
        except Exception as e:
            print(f"Errore durante la copia di '{nome_elemento}': {e}")
            
    print("\nCopia completata con successo!")

# --- Esempio di utilizzo ---
if __name__ == "__main__":
    # Download latest version
    cartella_sorgente = kagglehub.dataset_download("hammaadali/insects-recognition")

    print("Path to dataset files:", cartella_sorgente)
    # Sostituisci con i tuoi percorsi reali
    cartella_destinazione = "./dataset"
    
    # Esegui la funzione di copia
    copia_contenuto_cartella(cartella_sorgente, cartella_destinazione)




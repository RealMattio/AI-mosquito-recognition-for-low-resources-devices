import os
import shutil
import random

def dividi_e_accorpa_dataset(percorso_sorgente, percorso_destinazione, nome_cartella_mosquito="Mosquito", train_ratio=0.7, val_ratio=0.2):
    """
    Legge una cartella con diverse specie di insetti. 
    Mantiene la classe 'Mosquito' e accorpa tutte le altre in 'Not_Mosquito'.
    Infine divide tutto in Train, Validation e Test.
    """
    # Controlli iniziali di sicurezza
    if not os.path.exists(percorso_sorgente):
        print(f"Errore: Il percorso sorgente '{percorso_sorgente}' non esiste.")
        return

    if train_ratio + val_ratio >= 1.0:
        print("Errore: La somma di train_ratio e val_ratio deve essere minore di 1.")
        return

    # Se la cartella di destinazione esiste già, la cancelliamo per fare un lavoro pulito
    if os.path.exists(percorso_destinazione):
        shutil.rmtree(percorso_destinazione)
        print(f"Cartella di destinazione '{percorso_destinazione}' esistente rimossa.")

    # 1. Creiamo la nuova struttura di base
    fasi = ['train', 'validation', 'test']
    classi_nuove = ['Mosquito', 'Not_Mosquito']
    
    for fase in fasi:
        for classe in classi_nuove:
            os.makedirs(os.path.join(percorso_destinazione, fase, classe))
            
    print(f"Creata la struttura delle cartelle in '{percorso_destinazione}'\n")

    # 2. Raccogliamo i percorsi di tutte le immagini nelle due macro-categorie
    immagini_mosquito = []
    immagini_not_mosquito = []

    print("Analisi delle cartelle sorgenti in corso...")
    for nome_cartella in os.listdir(percorso_sorgente):
        path_cartella = os.path.join(percorso_sorgente, nome_cartella)
        
        if os.path.isdir(path_cartella):
            # Troviamo tutte le immagini (ignorando eventuali sottocartelle nascoste)
            file_nella_cartella = [os.path.join(path_cartella, f) for f in os.listdir(path_cartella) if os.path.isfile(os.path.join(path_cartella, f))]
            
            # Dividiamo la logica: è la cartella delle zanzare o è un altro insetto?
            # Usiamo .lower() così non abbiamo problemi se c'è scritto "mosquito" o "Mosquito"
            if nome_cartella.lower() == nome_cartella_mosquito.lower():
                immagini_mosquito.extend(file_nella_cartella)
                print(f" -> Trovate {len(file_nella_cartella)} immagini di Zanzare (cartella '{nome_cartella}').")
            else:
                immagini_not_mosquito.extend(file_nella_cartella)
                print(f" -> Trovate {len(file_nella_cartella)} immagini di ALTRI insetti (cartella '{nome_cartella}'). Verranno accorpate.")

    print(f"\nRiepilogo finale prima della divisione:")
    print(f"Totale Mosquito: {len(immagini_mosquito)}")
    print(f"Totale Not_Mosquito (tutte le altre specie accorpate): {len(immagini_not_mosquito)}\n")

    # 3. Funzione interna per mescolare, tagliare e copiare i file
    def smista_e_copia(lista_immagini, nome_classe_destinazione):
        # Mescoliamo le immagini per garantire casualità
        random.shuffle(lista_immagini) 
        
        # Calcoliamo dove fare i "tagli" nella lista
        n_totale = len(lista_immagini)
        idx_train = int(n_totale * train_ratio)
        idx_val = idx_train + int(n_totale * val_ratio)
        
        # Tagliamo la lista
        train_files = lista_immagini[:idx_train]
        val_files = lista_immagini[idx_train:idx_val]
        test_files = lista_immagini[idx_val:]
        
        # Copiamo fisicamente i file nelle nuove cartelle
        # Usiamo os.path.basename per mantenere il nome originale del file (.jpg, .png)
        for f in train_files:
            shutil.copy(f, os.path.join(percorso_destinazione, 'train', nome_classe_destinazione, os.path.basename(f)))
        for f in val_files:
            shutil.copy(f, os.path.join(percorso_destinazione, 'validation', nome_classe_destinazione, os.path.basename(f)))
        for f in test_files:
            shutil.copy(f, os.path.join(percorso_destinazione, 'test', nome_classe_destinazione, os.path.basename(f)))
            
        print(f"Classe {nome_classe_destinazione} creata: {len(train_files)} Train | {len(val_files)} Validation | {len(test_files)} Test")

    # 4. Eseguiamo l'effettiva divisione e copia
    print("Inizio la copia dei file (potrebbe volerci qualche istante)...")
    smista_e_copia(immagini_mosquito, 'Mosquito')
    smista_e_copia(immagini_not_mosquito, 'Not_Mosquito')

    print("\nOperazione completata con successo! 🎉")


if __name__ == '__main__':
    # ==========================================
    # CONFIGURAZIONE - Modifica questi valori!
    # ==========================================
    
    # 1. Il percorso della cartella che contiene tutte le specie di insetti
    percorso_originale = './dataset' 
    
    # 2. Il percorso dove vuoi creare il nuovo dataset diviso
    percorso_nuovo = './dataset_splitted'
    
    # 3. Come si chiama ESATTAMENTE la cartella delle zanzare nel dataset originale?
    nome_cartella_zanzare = 'Mosquito' 
    
    # Avvia la magia!
    dividi_e_accorpa_dataset(percorso_originale, percorso_nuovo, nome_cartella_zanzare, train_ratio=0.7, val_ratio=0.2)
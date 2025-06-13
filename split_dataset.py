import os
import shutil
import random
import sys

def dividi_dataset(percorso_sorgente, percorso_destinazione, train_ratio=0.7, val_ratio=0.2):
    """
    Divide un dataset di immagini in cartelle train, validation e test.

    La struttura della cartella sorgente deve essere:
    percorso_sorgente/
    ├── classe_1/
    │   ├── img1.jpg
    │   └── ...
    └── classe_2/
        ├── img2.jpg
        └── ...

    La funzione creerà la seguente struttura nella destinazione:
    percorso_destinazione/
    ├── train/
    │   ├── classe_1/
    │   └── classe_2/
    ├── validation/
    │   ├── classe_1/
    │   └── classe_2/
    └── test/
        ├── classe_1/
        └── classe_2/

    Args:
        percorso_sorgente (str): Percorso della cartella con i dati originali.
        percorso_destinazione (str): Percorso dove salvare il nuovo dataset.
        train_ratio (float): Percentuale di immagini per il training set (es. 0.7).
        val_ratio (float): Percentuale di immagini per il validation set (es. 0.2).
                         Il test set conterrà le immagini rimanenti.
    """
    # Controlla se il percorso sorgente esiste
    if not os.path.exists(percorso_sorgente):
        print(f"Errore: Il percorso sorgente '{percorso_sorgente}' non esiste.")
        return

    # Percentuale per il test set calcolata automaticamente
    test_ratio = 1 - train_ratio - val_ratio
    if not (train_ratio + val_ratio + test_ratio) == 1.0:
        print("Errore: La somma delle percentuali deve essere 1.")
        return

    # Rimuove la cartella di destinazione se esiste già per evitare errori
    if os.path.exists(percorso_destinazione):
        shutil.rmtree(percorso_destinazione)
        print(f"Cartella di destinazione '{percorso_destinazione}' esistente rimossa.")

    # Crea le cartelle train, validation e test
    path_train = os.path.join(percorso_destinazione, 'train')
    path_val = os.path.join(percorso_destinazione, 'validation')
    path_test = os.path.join(percorso_destinazione, 'test')

    os.makedirs(path_train)
    os.makedirs(path_val)
    os.makedirs(path_test)
    print(f"Create le cartelle: train, validation, test in '{percorso_destinazione}'")

    # Itera su ogni classe presente nella cartella sorgente
    for nome_classe in os.listdir(percorso_sorgente):
        path_classe_sorgente = os.path.join(percorso_sorgente, nome_classe)
        
        if os.path.isdir(path_classe_sorgente):
            print(f"\nProcessing classe: {nome_classe}")
            
            # Crea le sottocartelle per la classe corrente in train, validation e test
            os.makedirs(os.path.join(path_train, nome_classe))
            os.makedirs(os.path.join(path_val, nome_classe))
            os.makedirs(os.path.join(path_test, nome_classe))

            # Ottieni la lista di tutte le immagini per la classe corrente
            immagini = [f for f in os.listdir(path_classe_sorgente) if os.path.isfile(os.path.join(path_classe_sorgente, f))]
            
            # Mescola la lista di immagini in modo casuale
            random.shuffle(immagini)

            # Calcola gli indici per la divisione
            n_immagini_totali = len(immagini)
            indice_train = int(n_immagini_totali * train_ratio)
            indice_val = indice_train + int(n_immagini_totali * val_ratio)

            # Divide la lista di immagini
            immagini_train = immagini[:indice_train]
            immagini_val = immagini[indice_train:indice_val]
            immagini_test = immagini[indice_val:] # Le rimanenti vanno nel test

            # Funzione di supporto per copiare i file
            def copia_file(lista_file, cartella_dest):
                for nome_file in lista_file:
                    sorgente = os.path.join(path_classe_sorgente, nome_file)
                    destinazione = os.path.join(cartella_dest, nome_classe, nome_file)
                    shutil.copy(sorgente, destinazione)

            # Copia i file nelle rispettive cartelle di destinazione
            copia_file(immagini_train, path_train)
            copia_file(immagini_val, path_val)
            copia_file(immagini_test, path_test)

            print(f"  - {len(immagini_train)} immagini in 'train/{nome_classe}'")
            print(f"  - {len(immagini_val)} immagini in 'validation/{nome_classe}'")
            print(f"  - {len(immagini_test)} immagini in 'test/{nome_classe}'")

    print("\nDivisione del dataset completata con successo! 🎉")


if __name__ == '__main__':
    # --- CONFIGURAZIONE ---
    # Sostituisci con il percorso della tua cartella di immagini
    percorso_sorgente = 'augmented_dataset'  # Cartella con le immagini originali
    # Sostituisci con il percorso dove vuoi salvare il nuovo dataset
    percorso_destinazione = 'augmented_dataset_splitted'

    # Crea una finta cartella sorgente per un test rapido
    if not os.path.exists(percorso_sorgente):
        print("Creazione di una cartella di test 'dataset_originale'...")
        os.makedirs(os.path.join(percorso_sorgente, 'cani'))
        os.makedirs(os.path.join(percorso_sorgente, 'gatti'))
        for i in range(100): # 100 immagini per classe
            open(os.path.join(percorso_sorgente, 'cani', f'cane_{i}.jpg'), 'a').close()
            open(os.path.join(percorso_sorgente, 'gatti', f'gatto_{i}.jpg'), 'a').close()
        print("Cartella di test creata.")

    # Esegui la funzione con le percentuali 70/20/10
    dividi_dataset(percorso_sorgente, percorso_destinazione, train_ratio=0.7, val_ratio=0.2)
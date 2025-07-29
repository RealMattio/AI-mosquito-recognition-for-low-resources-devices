import os
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image

def analizza_dataset(percorso_dataset, genera_grafico=True):
    """
    Analizza un dataset di immagini e ne stampa le statistiche.

    La struttura della cartella sorgente deve essere:
    percorso_dataset/
    ├── classe_1/
    │   ├── img1.jpg
    │   └── ...
    └── classe_2/
        ├── img2.jpg
        └── ...

    Args:
        percorso_dataset (str): Percorso della cartella contenente il dataset.
        genera_grafico (bool): Se True, genera e salva un grafico a barre
                               della distribuzione delle classi.
    """
    # 1. Controlla che il percorso sia una cartella valida
    if not os.path.isdir(percorso_dataset):
        print(f"Errore: Il percorso '{percorso_dataset}' non è una cartella valida.")
        return

    # 2. Ottieni i nomi delle classi (sottocartelle)
    nomi_classi = [d for d in os.listdir(percorso_dataset) if os.path.isdir(os.path.join(percorso_dataset, d))]
    num_classi = len(nomi_classi)

    if num_classi == 0:
        print(f"Nessuna sottocartella (classe) trovata in '{percorso_dataset}'.")
        return

    # 3. Inizializza le variabili per le statistiche
    immagini_per_classe = {}
    contatore_dimensioni = Counter()
    contatore_formati = Counter()
    immagini_totali = 0
    immagini_corrotte = 0

    print(f"🔎 Analisi del dataset in corso: '{os.path.abspath(percorso_dataset)}'...")

    # 4. Itera su ogni classe per raccogliere i dati
    for nome_classe in nomi_classi:
        path_classe = os.path.join(percorso_dataset, nome_classe)
        files_nella_classe = os.listdir(path_classe)
        immagini_per_classe[nome_classe] = len(files_nella_classe)
        immagini_totali += len(files_nella_classe)

        # Analizza ogni file per ottenere dimensioni e formato
        for nome_file in files_nella_classe:
            percorso_file = os.path.join(path_classe, nome_file)
            
            # Ottieni formato dall'estensione
            estensione = os.path.splitext(nome_file)[1].lower()
            if estensione:
                contatore_formati[estensione] += 1
            
            # Ottieni dimensioni con Pillow
            try:
                with Image.open(percorso_file) as img:
                    contatore_dimensioni[img.size] += 1
            except (IOError, SyntaxError):
                immagini_corrotte += 1


    # 5. Stampa le statistiche raccolte
    print("\n--- 📊 Statistiche del Dataset ---\n")
    print(f"▪️ **Numero totale di classi**: {num_classi}")
    print(f"▪️ **Numero totale di immagini**: {immagini_totali}")
    if immagini_corrotte > 0:
        print(f"▪️ **Immagini non leggibili/corrotte**: {immagini_corrotte}")

    print("\n--- 🗂️ Distribuzione per Classe ---")
    classi_ordinate = sorted(immagini_per_classe.items(), key=lambda item: item[1], reverse=True)
    for nome, conteggio in classi_ordinate:
        percentuale = (conteggio / immagini_totali) * 100
        print(f"  - **{nome}**: {conteggio} immagini ({percentuale:.2f}%)")

    print("\n--- 🖼️ Formati delle Immagini ---")
    for formato, conteggio in contatore_formati.most_common():
        print(f"  - **{formato}**: {conteggio} file")

    print("\n--- 📏 Dimensioni (larghezza x altezza) ---")
    if not contatore_dimensioni:
        print("  Impossibile leggere le dimensioni delle immagini.")
    else:
        print(f"  Trovate {len(contatore_dimensioni)} dimensioni uniche.")
        # Stampa le 5 dimensioni più comuni
        for (larghezza, altezza), conteggio in contatore_dimensioni.most_common(5):
            print(f"  - **{larghezza}x{altezza}**: {conteggio} immagini")
        if len(contatore_dimensioni) > 5:
            print("  ...")

    # 6. Genera il grafico se richiesto
    if genera_grafico and immagini_per_classe:
        crea_grafico_distribuzione(immagini_per_classe, percorso_dataset)


def crea_grafico_distribuzione(immagini_per_classe, percorso_output):
    """Genera e salva un grafico a barre della distribuzione delle classi."""
    nomi_classi = list(immagini_per_classe.keys())
    conteggi = list(immagini_per_classe.values())

    plt.figure(figsize=(12, 7))
    barre = plt.bar(nomi_classi, conteggi, color='dodgerblue', alpha=0.9)
    plt.ylabel('Numero di Immagini', fontsize=12)
    plt.title('Distribuzione delle Immagini per Classe', fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Aggiunge il numero esatto sopra ogni barra
    for barra in barre:
        altezza = barra.get_height()
        plt.text(barra.get_x() + barra.get_width()/2.0, altezza, f'{altezza}', ha='center', va='bottom')

    nome_file_grafico = 'distribuzione_classi.png'
    percorso_salvataggio = os.path.join(percorso_output, nome_file_grafico)
    plt.savefig(percorso_salvataggio)
    print(f"\n✅ Grafico della distribuzione salvato in: '{percorso_salvataggio}'")


if __name__ == '__main__':
    # --- CONFIGURAZIONE ---
    # Sostituisci questo con il percorso della cartella del tuo dataset
    percorso_cartella_dataset = 'augmented_dataset_splitted/test'  # Esempio di percorso

    # Crea una finta cartella di test se non esiste
    if not os.path.exists(percorso_cartella_dataset):
        print(f"La cartella '{percorso_cartella_dataset}' non esiste. Ne creo una di esempio.")
        os.makedirs(os.path.join(percorso_cartella_dataset, 'cani'))
        os.makedirs(os.path.join(percorso_cartella_dataset, 'gatti'))
        # Crea finti file immagine
        for i in range(70):
            open(os.path.join(percorso_cartella_dataset, 'cani', f'cane_{i}.jpg'), 'a').close()
        for i in range(100):
            open(os.path.join(percorso_cartella_dataset, 'gatti', f'gatto_{i}.png'), 'a').close()

    # Esegui l'analisi
    analizza_dataset(percorso_cartella_dataset)
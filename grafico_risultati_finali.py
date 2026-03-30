import matplotlib.pyplot as plt
import json

def leggi_file_json(percorso_file):
    """
    Questa funzione apre un file JSON e ne restituisce il contenuto.
    
    Parametri:
    percorso_file (str): Una stringa che contiene il percorso del file (es. "dati.json").
    
    Ritorna:
    dict: Il contenuto del file sotto forma di dizionario.
    """
    try:
        # Apriamo il file in modalità lettura ('r')
        with open(percorso_file, 'r', encoding='utf-8') as file:
            # Carichiamo i dati dal file e trasformiamoli in un dizionario
            dati = json.load(file)
            return dati
            
    except FileNotFoundError:
        return "Errore: Il file non è stato trovato. Controlla il percorso!"
    except json.JSONDecodeError:
        return "Errore: Il file non sembra essere un JSON valido."
    except Exception as e:
        return f"Si è verificato un errore imprevisto: {e}"

def crea_grafici_paper(dataset):
    """
    Estrae i dati dal dizionario e genera due grafici affiancati per un paper scientifico.
    """
    
    # 1. Definiamo le chiavi per navigare il dizionario
    livelli_aug = ["aug0", "aug25", "aug50", "aug75", "aug100"]
    etichette_asse_x = ["100%", "225%", "350%", "475%", "600%"] # Più belle da leggere
    modelli = ["MobileNet", "MobileNetV2", "NASNetMobile", "ResNet50"]
    x_label = "Percentage original dataset"
    y_label = "F1-Score (Mosquito Class)"
    
    # Inizializziamo i contenitori per i dati
    dati_unfreeze = {modello: [] for modello in modelli}
    dati_freeze = {modello: [] for modello in modelli}
    
    # 2. Estrazione dei dati
    for aug in livelli_aug:
        for modello in modelli:
            # Estraiamo per i modelli UNFREEZE
            chiave_unfreeze = f"{modello}_unfreeze"
            # Nota: uso 'macro avg'. Se vuoi l'F1 della sola classe zanzara, 
            # cambia 'macro avg' in 'Mosquito'
            f1_unf = dataset[aug][chiave_unfreeze]["classification_report"]["Mosquito"]["f1-score"]
            dati_unfreeze[modello].append(f1_unf)
            
            # Estraiamo per i modelli FREEZE
            chiave_freeze = f"{modello}_freeze"
            f1_frz = dataset[aug][chiave_freeze]["classification_report"]["Mosquito"]["f1-score"]
            dati_freeze[modello].append(f1_frz)

    # 3. Impostazioni grafiche per il PAPER SCIENTIFICO
    # Aumentiamo enormemente i font per garantire la leggibilità
    plt.rcParams.update({
        'font.size': 18,           # Dimensione base del testo
        'axes.labelsize': 20,      # Etichette assi (X, Y)
        'axes.titlesize': 22,      # Titoli dei singoli grafici
        'xtick.labelsize': 16,     # Numeri sull'asse X
        'ytick.labelsize': 16,     # Numeri sull'asse Y
        'legend.fontsize': 16      # Testo della legenda
    })
    
    # Mappa rigida per colori e marker (garantisce coerenza visiva)
    stile_linee = {
        "MobileNet":    {"color": "#1f77b4", "marker": "o"}, # Blu, pallino
        "MobileNetV2":  {"color": "#ff7f0e", "marker": "s"}, # Arancione, quadrato
        "NASNetMobile": {"color": "#2ca02c", "marker": "^"}, # Verde, triangolo
        "ResNet50":     {"color": "#d62728", "marker": "D"}  # Rosso, rombo
    }

    # 4. Creazione della figura (sharey=True mantiene la stessa scala verticale!)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    
    # 5. Tracciamento delle linee
    for modello in modelli:
        # Grafico di Sinistra (Unfreeze)
        ax1.plot(etichette_asse_x, dati_unfreeze[modello], 
                 label=modello, linewidth=3, markersize=10, **stile_linee[modello])
        
        # Grafico di Destra (Freeze)
        ax2.plot(etichette_asse_x, dati_freeze[modello], 
                 label=modello, linewidth=3, markersize=10, **stile_linee[modello])

    # 6. Personalizzazione e Titoli
    ax1.set_title("Unfreeze Models")
    ax2.set_title("Freeze Models")
    
    ax1.set_xlabel(x_label)
    ax2.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    
    # Aggiunta di una griglia leggera per aiutare l'occhio
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Aggiungiamo la legenda al primo grafico (o potresti metterla esterna)
    ax1.legend(loc='lower right')
    ax2.legend(loc='lower right')

    # Ottimizza gli spazi bianchi in modo che non ci siano sovrapposizioni
    plt.tight_layout()
    
    # Mostra e salva il grafico (puoi decommentare l'ultima riga per salvare)
    #plt.show()
    plt.savefig("grafici/f1_score_VS_augmentation.png", dpi=300, bbox_inches='tight')

# Esegui la funzione
# crea_grafici_paper(dataset)

if __name__ == "__main__":
    # Carica il dataset da un file JSON (modifica il percorso se necessario)
    dataset = leggi_file_json("risultati_finali.json")
    
    # Crea i grafici per il paper scientifico
    crea_grafici_paper(dataset)
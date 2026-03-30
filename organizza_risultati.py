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

# --- Esempio di utilizzo ---
# miodizionario = leggi_file_json("mio_file.json")
# print(miodizionario)


def inserisci_dizionario(dizionario_da_inserire, dizionario_destinazione, chiave_stringa):
    """
    Inserisce un dizionario dentro un altro dizionario usando una chiave specifica.
    
    Parametri:
    dizionario_da_inserire (dict): Il dizionario che vogliamo "traslocare".
    dizionario_destinazione (dict): Il dizionario che riceverà i dati.
    chiave_stringa (str): Il nome della chiave a cui associare il primo dizionario.
    
    Ritorna:
    dict: Il dizionario di destinazione aggiornato.
    """
    
    # Assegniamo il primo dizionario come valore della chiave specificata
    dizionario_destinazione[chiave_stringa] = dizionario_da_inserire
    
    return dizionario_destinazione

# --- Esempio pratico di utilizzo ---
# dati_utente = {"nome": "Luca", "città": "Roma"}
# database_totale = {}
# identificativo = "utente_01"

# risultato = inserisci_dizionario(dati_utente, database_totale, identificativo)
# print(risultato)
# Output previsto: {'utente_01': {'nome': 'Luca', 'città': 'Roma'}}


nomi_files = ["evaluation_metrics_MobileNet_ClassicDenseClassifier.json", "evaluation_metrics_MobileNetV2_ClassicDenseClassifier.json", 
                "evaluation_metrics_NASNetMobile_ClassicDenseClassifier.json", "evaluation_metrics_ResNet50_ClassicDenseClassifier.json"]

risultati_finali = {}

cartelle_history = [
    "keras_models_160326_False_unfreeze_False_augmented_performances_and_history", #freeze 0
    "keras_models_160326_True_unfreeze_False_augmented_performances_and_history", #unfreeze 0
    "keras_models_170326-0957_False_unfreeze_aug25pct_performances_and_history", #freeze 25
    "keras_models_180326-1012_True_unfreeze_aug25pct_performances_and_history", #unfreeze 25
    "keras_models_170326-1346_False_unfreeze_aug50pct_performances_and_history", #freeze 50
    "keras_models_180326-1200_True_unfreeze_aug50pct_performances_and_history", #unfreeze 50
    "keras_models_170326-1620_False_unfreeze_aug75pct_performances_and_history", #freeze 75
    "keras_models_180326-1544_True_unfreeze_aug75pct_performances_and_history", #unfreeze 75
    "keras_models_1108_performances_and_history", #freeze 100
    "keras_models_1503_True_unfreeze_True_augmented_performances_and_history", #unfreeze 100

]

augmentation = ['aug0', 'aug25', 'aug50', 'aug75', 'aug100']
freeze = ['unfreeze', 'freeze']
i = 0
for aug in augmentation:
    ris = {}
    cartella_freeze = f"keras_training/{cartelle_history[i]}"
    i += 1
    for file in nomi_files:
        percorso_file = f"{cartella_freeze}/{file}"
        dati = leggi_file_json(percorso_file)
        modello = file.split("_")[2]  # Estrae il nome del modello dal nome del file
        ris[f"{modello}_{freeze[i%2]}"] = dati
    
    cartella_unfreeze = f"keras_training/{cartelle_history[i]}"
    i += 1
    for file in nomi_files:
        percorso_file = f"{cartella_unfreeze}/{file}"
        dati = leggi_file_json(percorso_file)
        modello = file.split("_")[2]  # Estrae il nome del modello dal nome del file
        ris[f"{modello}_{freeze[i%2]}"] = dati
    risultati_finali[f"{aug}"] = ris

with open("risultati_finali.json", "w", encoding='utf-8') as file_output:
    json.dump(risultati_finali, file_output, indent=4, ensure_ascii=False)

print("I risultati finali sono stati salvati in 'risultati_finali.json'")
#print(risultati_finali)
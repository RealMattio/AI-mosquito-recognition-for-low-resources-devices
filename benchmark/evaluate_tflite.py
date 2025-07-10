# evaluate_tflite.py
# Data: 10 luglio 2025

import tensorflow as tf
import numpy as np
import os
import json
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

def evaluate_tflite_model(model_path, dataset_dir, output_dir):
    """
    Carica un modello TFLite, lo valuta su un dataset di test e salva
    un report completo con metriche e grafici.
    """
    print(f"--- Inizio Valutazione Modello: {os.path.basename(model_path)} ---")
    
    # --- 1. Caricamento del Modello TFLite ---
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        print("Modello TFLite caricato con successo.")
    except Exception as e:
        print(f"ERRORE: Impossibile caricare il file del modello TFLite: {e}")
        return

    # --- 2. Caricamento del Dataset di Test ---
    # Ottiene la dimensione di input richiesta dal modello
    height = input_details['shape'][1]
    width = input_details['shape'][2]
    
    print(f"Caricamento dataset da '{dataset_dir}' con dimensione input: ({height}, {width})")
    try:
        test_ds = tf.keras.utils.image_dataset_from_directory(
            dataset_dir,
            label_mode='int',
            seed=123,
            image_size=(height, width),
            batch_size=32, # Un batch size ragionevole per l'inferenza
            shuffle=False # MAI mescolare il set di test
        )
        class_names = test_ds.class_names
        print(f"Classi trovate: {class_names}")
    except Exception as e:
        print(f"ERRORE: Impossibile caricare il dataset: {e}")
        return

    # --- 3. Esecuzione dell'Inferenza su tutto il Dataset ---
    all_predictions = []
    all_true_labels = []

    print("Esecuzione inferenza su tutto il set di test...")
    # Itera su ogni batch di immagini e etichette nel dataset
    for images, labels in test_ds:
        for image in images:
            # Aggiungi la dimensione del batch (da H,W,C a 1,H,W,C)
            img_batch = np.expand_dims(image, axis=0).astype(input_details['dtype'])
            
            # Imposta il tensore di input
            interpreter.set_tensor(input_details['index'], img_batch)
            
            # Esegui l'inferenza
            interpreter.invoke()
            
            # Ottieni il risultato e aggiungilo alla lista
            output_data = interpreter.get_tensor(output_details['index'])
            all_predictions.append(output_data[0])

        # Aggiungi le etichette reali alla lista
        all_true_labels.extend(labels.numpy())
    
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    
    print("Inferenza completata.")
    
    # --- 4. Calcolo delle Metriche ---
    print("Calcolo delle metriche di classificazione...")
    
    # Ottieni le classi predette (l'indice del valore massimo)
    y_pred = np.argmax(all_predictions, axis=1)
    
    # Calcola il classification report come dizionario
    report_dict = classification_report(all_true_labels, y_pred, target_names=class_names, output_dict=True)
    
    # --- 5. Creazione e Salvataggio dei Grafici ---
    os.makedirs(output_dir, exist_ok=True)
    model_base_name = os.path.splitext(os.path.basename(model_path))[0]

    # Matrice di Confusione
    cm = confusion_matrix(all_true_labels, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Matrice di Confusione - {model_base_name}')
    plt.ylabel('Etichetta Reale')
    plt.xlabel('Etichetta Predetta')
    cm_path = os.path.join(output_dir, f'confusion_matrix_{model_base_name}.png')
    plt.savefig(cm_path)
    plt.close()
    print(f"Matrice di confusione salvata in: {cm_path}")

    # Curva ROC e AUC (per classificazione binaria)
    roc_auc = None
    roc_path = None
    if len(class_names) == 2:
        # Probabilità per la classe positiva (classe 1)
        y_prob = all_predictions[:, 1]
        fpr, tpr, _ = roc_curve(all_true_labels, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Curva ROC - {model_base_name}')
        plt.legend(loc="lower right")
        roc_path = os.path.join(output_dir, f'roc_curve_{model_base_name}.png')
        plt.savefig(roc_path)
        plt.close()
        print(f"Curva ROC salvata in: {roc_path}")

    # --- 6. Creazione e Salvataggio del File JSON ---
    final_report = {
        "classification_report": report_dict,
        "confusion_matrix_path": cm_path,
        "roc_auc": roc_auc,
        "roc_curve_path": roc_path
    }
    
    json_path = os.path.join(output_dir, f'evaluation_{model_base_name}.json')
    with open(json_path, 'w') as f:
        json.dump(final_report, f, indent=4)
        
    print(f"Report di valutazione completo salvato in: {json_path}")
    print("--- Valutazione Terminata ---")


if __name__ == '__main__':
    # Configura gli argomenti da riga di comando
    parser = argparse.ArgumentParser(description="Valuta un modello TFLite su un dataset di test.")
    
    parser.add_argument(
        '--model_path', 
        type=str, 
        required=True,
        help="Il percorso completo al file del modello .tflite da valutare."
    )
    parser.add_argument(
        '--dataset_dir', 
        type=str,
        default='augmented_dataset_splitted/test',
        help="Il percorso alla cartella del dataset di test, con sottocartelle per ogni classe."
    )
    parser.add_argument(
        '--output_dir', 
        type=str, 
        required=True,
        help="La cartella dove verranno salvati i risultati (report JSON e grafici)."
    )
    
    args = parser.parse_args()
    
    # Esegui la funzione di valutazione
    evaluate_tflite_model(args.model_path, args.dataset_dir, args.output_dir)

# Esempio di utilizzo:
# python evaluate_tflite.py --model_path path/to/model.tflite --dataset_dir path/to/test_dataset --output_dir path/to/output
# python evaluate_tflite.py --model_path models/efficientnet_lite0.tflite --output_dir evaluation_results
import os
import json
import argparse
from typing import Dict, Any, Union

def round_values_recursively(data: Any) -> Any:
    """
    Scansiona ricorsivamente una struttura dati (dizionario o lista)
    e arrotonda tutti i valori di tipo float a due cifre decimali.
    """
    if isinstance(data, dict):
        # Se è un dizionario, applica la ricorsione a ogni suo valore
        return {key: round_values_recursively(value) for key, value in data.items()}
    elif isinstance(data, list):
         # Se è una lista, applica la ricorsione a ogni suo elemento
        return [round_values_recursively(item) for item in data]
    elif isinstance(data, float):
        # Se è un float, arrotondalo a 4 cifre decimali
        return round(data, 4)
    else:
        # Per tutti gli altri tipi (int, str, etc.), restituisci il valore com'è
        return data

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calcola la variazione percentuale tra un vecchio e un nuovo valore.
    Gestisce la divisione per zero.
    """
    if old_value == 0:
        return 0.0 if new_value == 0 else float('inf')
    
    change = ((new_value - old_value) / old_value) * 100
    return round(change, 2)

def process_metrics(old_data: Dict[str, Any], new_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Scansiona ricorsivamente i dizionari di metriche e calcola la variazione percentuale
    per ogni valore numerico trovato.
    """
    change_report = {}

    for key, new_value in new_data.items():
        if key not in old_data:
            change_report[key] = new_value
            continue

        old_value = old_data[key]

        if isinstance(new_value, dict) and isinstance(old_value, dict):
            change_report[key] = process_metrics(old_value, new_value)
        elif isinstance(new_value, (int, float)) and isinstance(old_value, (int, float)):
            change_report[key] = calculate_percentage_change(old_value, new_value)
        else:
            change_report[key] = new_value
            
    return change_report

def main(before_file: str, after_file: str, output_file: str):
    """
    Funzione principale che confronta due file JSON e salva le variazioni.
    """
    print(f"File 'prima': {before_file}")
    print(f"File 'dopo': {after_file}")
    print(f"File di output: {output_file}\n")

    for f_path in [before_file, after_file]:
        if not os.path.exists(f_path):
            print(f"ERRORE: Il file di input '{f_path}' non è stato trovato.")
            return

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(before_file, 'r') as f:
        before_data = json.load(f)
    
    with open(after_file, 'r') as f:
        after_data = json.load(f)
    
    # >>> INIZIO MODIFICA <<<
    # Arrotonda tutti i valori float a 2 cifre decimali prima del calcolo
    print("Arrotondamento dei valori a 2 cifre decimali...")
    rounded_before_data = round_values_recursively(before_data)
    rounded_after_data = round_values_recursively(after_data)
    # >>> FINE MODIFICA <<<

    # Calcola le variazioni percentuali usando i dati già arrotondati
    percentage_change_data = process_metrics(rounded_before_data, rounded_after_data)
    
    with open(output_file, 'w') as f:
        json.dump(percentage_change_data, f, indent=4)
    
    print(f"Risultato salvato con successo in: '{output_file}'")
    print("Processo completato.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calcola la variazione percentuale delle performance tra due file JSON, arrotondando i valori prima del calcolo."
    )
    parser.add_argument(
        "before_file", 
        type=str, 
        help="Percorso del file JSON con i dati prima della modifica."
    )
    parser.add_argument(
        "after_file", 
        type=str, 
        help="Percorso del file JSON con i dati dopo la modifica."
    )
    parser.add_argument(
        "output_file", 
        type=str, 
        help="Percorso completo del file JSON dove salvare i risultati."
    )

    args = parser.parse_args()
    main(args.before_file, args.after_file, args.output_file)


    # esempio

    # python calculate_performance_variation.py before.json after.json output.json
    # python calculate_performance_variation.py keras_training/keras_models_1503_True_unfreeze_True_augmented_performances_and_history/evaluation_metrics_MobileNet_ClassicDenseClassifier.json tflite_models/tflite_models_full_trained/evaluation_MobileNet_full_quant_int8.json tflite_models/tflite_models_full_trained/variation_MobileNet_full_quant_int8.json
    # python calculate_performance_variation.py keras_training/keras_models_1503_True_unfreeze_True_augmented_performances_and_history/evaluation_metrics_MobileNetV2_ClassicDenseClassifier.json tflite_models/tflite_models_full_trained/evaluation_MobileNetV2_full_quant_int8.json tflite_models/tflite_models_full_trained/variation_MobileNetV2_full_quant_int8.json ; python calculate_performance_variation.py keras_training/keras_models_1503_True_unfreeze_True_augmented_performances_and_history/evaluation_metrics_NASNetMobile_ClassicDenseClassifier.json tflite_models/tflite_models_full_trained/evaluation_NASNetMobile_full_quant_int8.json tflite_models/tflite_models_full_trained/variation_NASNetMobile_full_quant_int8.json ; python calculate_performance_variation.py keras_training/keras_models_1503_True_unfreeze_True_augmented_performances_and_history/evaluation_metrics_ResNet50_ClassicDenseClassifier.json tflite_models/tflite_models_full_trained/evaluation_ResNet50_full_quant_int8.json tflite_models/tflite_models_full_trained/variation_ResNet50_full_quant_int8.json
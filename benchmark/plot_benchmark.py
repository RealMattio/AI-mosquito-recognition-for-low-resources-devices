# plot_all_benchmarks.py
# Data: 26 agosto 2025
# Luogo: Latina, Lazio, Italia

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def clean_model_name(raw_name: str) -> str:
    """Semplifica i nomi lunghi dei modelli per renderli più leggibili sul grafico."""
    name = raw_name.replace('_ClassicDenseClassifier_final_model', '')
    
    if name.endswith('.keras_Keras'):
        return name.replace('.keras_Keras', ' (Keras)')
    elif name.endswith('.tflite_TFLite'):
        return name.replace('.tflite_TFLite', ' (TFLite)')
    else:
        return raw_name

def create_single_boxplot(ax, dataframe, title: str, palette: str):
    """
    Funzione helper per disegnare un singolo boxplot su un asse Matplotlib dato.
    Ordina i modelli per performance mediana e nasconde gli outlier.
    """
    # Rimuoviamo i tag (Keras/TFLite) per pulire le etichette dell'asse Y
    clean_df = dataframe.copy()
    clean_df['Modello'] = clean_df['Modello'].str.replace(' (TFLite)', '').str.replace(' (Keras)', '')

    # Ordina i modelli in base alla loro mediana
    sorted_models = clean_df.groupby('Modello')['Tempo di Inferenza (ms)'].median().sort_values().index
    
    sns.boxplot(
        ax=ax,
        data=clean_df,
        y='Modello',
        x='Tempo di Inferenza (ms)',
        order=sorted_models,
        orient='h',
        palette=palette,
        showfliers=False  # <-- MODIFICA CHIAVE: Nasconde gli outlier
    )
    ax.set_title(title, fontsize=20)
    ax.set_xlabel('Tempo di Inferenza (ms)', fontsize=20)
    ax.set_ylabel('Modello', fontsize=20)
    ax.xaxis.grid(True)
    ax.set(xlim=(0, None))

def create_all_benchmark_plots(json_path: str, output_prefix: str):
    """
    Carica i dati di benchmark e crea tre grafici distinti:
    1. Solo modelli Keras
    2. Solo modelli TFLite
    3. Keras e TFLite affiancati
    """
    print(f"Caricamento dei dati di benchmark da: {json_path}")
    
    # --- 1. Caricamento e Preparazione Dati ---
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERRORE: File non trovato in '{json_path}'.")
        return
        
    benchmark_data = data.get('benchmark_runs')
    if not benchmark_data:
        print("ERRORE: Il file JSON non contiene la chiave 'benchmark_runs'.")
        return

    plot_data = [{'Modello': clean_model_name(raw_name), 'Tempo di Inferenza (ms)': t}
                 for raw_name, results in benchmark_data.items()
                 for t in results.get('individual_times_ms', [])]
                 
    if not plot_data:
        print("Nessun dato di inferenza trovato.")
        return

    df = pd.DataFrame(plot_data)
    df_tflite = df[df['Modello'].str.contains('(TFLite)')]
    df_keras = df[df['Modello'].str.contains('(Keras)')]
    
    system_info = data.get('system_info', {})
    cpu_brand = system_info.get('cpu', {}).get('brand', 'Dispositivo Sconosciuto')
    main_title = f'Benchmark dei Tempi di Inferenza'
    #main_title = f'Benchmark dei Tempi di Inferenza\n(Dispositivo: {cpu_brand})'

    sns.set_theme(style="whitegrid")

    # --- 2. Creazione Grafico 1: Solo Keras ---
    print("Creazione grafico per i modelli Keras...")
    fig_keras, ax_keras = plt.subplots(figsize=(10, 5))
    #fig_keras.suptitle(main_title, fontsize=18)
    create_single_boxplot(ax_keras, df_keras, "Performance Modelli Keras (Originali)", "autumn")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    keras_output_path = f"{output_prefix}_keras.png"
    plt.savefig(keras_output_path, dpi=300)
    print(f"-> Grafico Keras salvato in: {keras_output_path}")
    plt.close(fig_keras)

    # --- 3. Creazione Grafico 2: Solo TFLite ---
    print("Creazione grafico per i modelli TFLite...")
    fig_tflite, ax_tflite = plt.subplots(figsize=(10, 5))
    #fig_tflite.suptitle(main_title, fontsize=18)
    create_single_boxplot(ax_tflite, df_tflite, "Performance Modelli TFLite (Quantizzati INT8)", "summer")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    tflite_output_path = f"{output_prefix}_tflite.png"
    plt.savefig(tflite_output_path, dpi=300)
    print(f"-> Grafico TFLite salvato in: {tflite_output_path}")
    plt.close(fig_tflite)

    # --- 4. Creazione Grafico 3: Combinato ---
    print("Creazione grafico combinato...")
    fig_combined, axes = plt.subplots(1, 2, figsize=(20, 7))
    fig_combined.suptitle(main_title, fontsize=20)
    
    create_single_boxplot(axes[1], df_keras, "Performance Modelli Keras (Originali)", "autumn")
    create_single_boxplot(axes[0], df_tflite, "Performance Modelli TFLite (Quantizzati INT8)", "summer")
    
    axes[1].set_ylabel('') # Rimuoviamo etichetta Y ridondante

    # <-- MODIFICA: Aumenta la dimensione dei font per gli indici (tick labels)
    axes[0].tick_params(axis='both', which='major', labelsize=20)
    axes[1].tick_params(axis='both', which='major', labelsize=20)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    combined_output_path = f"{output_prefix}_combined.png"
    plt.savefig(combined_output_path, dpi=300)
    print(f"-> Grafico combinato salvato in: {combined_output_path}")
    plt.close(fig_combined)
    
    print("\nSUCCESSO! Tutti i grafici sono stati generati.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Crea 3 grafici di benchmark (Keras, TFLite, Combinato) da un file JSON.")
    
    parser.add_argument(
        "json_file", 
        type=str,
        help="Il percorso al file JSON contenente i risultati del benchmark."
    )
    parser.add_argument(
        "--prefix", 
        type=str, 
        default="benchmark_results",
        help="Il prefisso per i nomi dei file immagine di output (default: benchmark_results)."
    )
    
    args = parser.parse_args()
    
    # Crea la cartella di output basata sul prefisso se non esiste
    output_dir = os.path.dirname(args.prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    create_all_benchmark_plots(args.json_file, args.prefix)

# Esempio di utilizzo:
# python benchmark/plot_benchmark.py benchmark/benchmark_results/benchmark_results_models_1108_on_WSL.json --prefix benchmark/benchmark_results_grafici/benchmark_results_WSL
# python benchmark/plot_benchmark.py benchmark/benchmark_results/benchmark_results_models_1108_on_RaspPi5.json --prefix benchmark/benchmark_results_grafici/benchmark_results_Pi5
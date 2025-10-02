# plot_cross_benchmark.py
# Data: 26 agosto 2025
# Luogo: Latina, Lazio, Italia

import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import argparse

def parse_model_key(raw_key: str) -> (str, str):
    """
    Estrae l'architettura e il formato del modello dal nome lungo del file.
    Restituisce una tupla (architettura, formato) o (None, None).
    """
    arch = None
    if 'MobileNetV2' in raw_key:
        arch = 'MobileNetV2'
    elif 'ResNet50' in raw_key:
        arch = 'ResNet50'
    elif 'NASNetMobile' in raw_key:
        arch = 'NASNetMobile'
    
    fmt = None
    if 'Keras' in raw_key:
        fmt = 'Keras'
    elif 'TFLite' in raw_key:
        fmt = 'TFLite'
        
    return arch, fmt

def create_cross_comparison_plot(wsl_json_path: str, pi_json_path: str, output_image_path: str):
    """
    Carica i dati da due file JSON e crea un grafico boxplot di confronto incrociato.
    """
    all_data = []
    
    # Processa entrambi i file JSON
    for path, platform_name in [(wsl_json_path, 'PC (WSL/GPU)'), (pi_json_path, 'Raspberry Pi (CPU)')]:
        print(f"Caricamento dati da: {path} per la piattaforma '{platform_name}'")
        try:
            with open(path, 'r') as f:
                data = json.load(f).get('benchmark_runs', {})
        except FileNotFoundError:
            print(f"ERRORE: File non trovato: {path}")
            continue

        # Estrae e formatta i dati
        for key, results in data.items():
            arch, fmt = parse_model_key(key)
            if arch and fmt:
                times = results.get('individual_times_ms', [])
                for t in times:
                    all_data.append({
                        'Architettura': arch,
                        'Piattaforma': platform_name,
                        'Formato': fmt,
                        'Tempo di Inferenza (ms)': t
                    })

    if not all_data:
        print("Nessun dato valido trovato nei file JSON. Impossibile creare il grafico.")
        return

    df = pd.DataFrame(all_data)
    
    # Filtra per includere solo i modelli richiesti
    df = df[df['Architettura'].isin(['MobileNetV2', 'ResNet50'])]

    print("Dati elaborati. Creazione del grafico in corso...")

    # --- Creazione del Grafico con Subplot ---
    sns.set_theme(style="whitegrid")
    # Crea una figura con 2 righe e 1 colonna
    fig, axes = plt.subplots(2, 1, figsize=(12, 14), sharex=True)
    
    # Titolo generale
    fig.suptitle('Confronto Performance Inferenza: Keras vs TFLite', fontsize=20, weight='bold')

    # Palette di colori e ordine per la legenda
    palette = {"Keras": "#d95f02", "TFLite": "#7570b3"}
    hue_order = ['Keras', 'TFLite']
    
    # Subplot 1: MobileNetV2
    ax1 = axes[0]
    sns.boxplot(ax=ax1, data=df[df['Architettura'] == 'MobileNetV2'],
                y='Piattaforma', x='Tempo di Inferenza (ms)', hue='Formato',
                hue_order=hue_order, palette=palette, showfliers=False)
    ax1.set_title('MobileNetV2', fontsize=16)
    ax1.set_ylabel('Piattaforma', fontsize=12)
    ax1.set_xlabel('') # Rimuoviamo l'etichetta x per non ripeterla
    ax1.set_xscale('log') # <-- Scala logaritmica per una migliore visualizzazione
    ax1.xaxis.set_major_formatter(mticker.ScalarFormatter()) # Formatta i tick per la leggibilità

    # Subplot 2: ResNet50
    ax2 = axes[1]
    sns.boxplot(ax=ax2, data=df[df['Architettura'] == 'ResNet50'],
                y='Piattaforma', x='Tempo di Inferenza (ms)', hue='Formato',
                hue_order=hue_order, palette=palette, showfliers=False)
    ax2.set_title('ResNet50', fontsize=16)
    ax2.set_ylabel('Piattaforma', fontsize=12)
    ax2.set_xlabel('Tempo di Inferenza (ms) - Scala Logaritmica', fontsize=12)
    ax2.set_xscale('log') # <-- Scala logaritmica
    ax2.xaxis.set_major_formatter(mticker.ScalarFormatter())

    # Migliora la legenda
    handles, labels = ax1.get_legend_handles_labels()
    ax1.get_legend().remove()
    ax2.get_legend().remove()
    fig.legend(handles, labels, title='Formato Modello', loc='upper right', bbox_to_anchor=(0.95, 0.92))

    plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adegua gli spazi per il titolo
    
    # --- Salvataggio della Figura ---
    try:
        plt.savefig(output_image_path, dpi=300)
        print(f"\nSUCCESSO! Grafico di confronto salvato in: {output_image_path}")
    except Exception as e:
        print(f"ERRORE: Impossibile salvare il grafico. Dettagli: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Crea un boxplot di confronto incrociato da due file JSON di benchmark.")
    
    parser.add_argument(
        "--wsl_json", 
        type=str,
        required=True,
        help="Il percorso al file JSON con i risultati del benchmark su WSL/PC."
    )
    parser.add_argument(
        "--pi_json", 
        type=str,
        required=True,
        help="Il percorso al file JSON con i risultati del benchmark su Raspberry Pi."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="benchmark_cross_platform_comparison.png",
        help="Il nome del file immagine di output per il grafico."
    )
    
    args = parser.parse_args()
    
    create_cross_comparison_plot(args.wsl_json, args.pi_json, args.output)


    # python benchmark/plot_cross_benchmark.py \
    # --wsl_json benchmark/benchmark_results/benchmark_results_models_1108_on_WSL.json \
    # --pi_json benchmark/benchmark_results/benchmark_results_models_1108_on_Pi5.json \
    # --output benchmark/benchmark_results/confronto_piattaforme.png


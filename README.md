# AI-mosquito-recognition-for-low-resources-devices

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.12-blue?logo=python)
![Static Badge](https://img.shields.io/badge/MicroPython-black?style=flat&logo=micropython&logoColor=white)
![Static Badge](https://img.shields.io/badge/Visual_Studio_Code-0078D4?style=flat&logo=visual%20studio%20code&logoColor=white)
![Static Badge](https://img.shields.io/badge/Arduino_IDE-00979D?style=flat&logoColor=white&logo=arduino)
![Static Badge](https://img.shields.io/badge/Mosquito🦟-friends-green)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/RealMattio/AI-mosquito-recognition-for-low-resources-devices?style=flat&label=Commit%20activity&color=yellow)

Pipeline completa di **classificazione binaria zanzara / non-zanzara** ottimizzata per dispositivi edge a risorse limitate (Raspberry Pi 5, Arduino Portenta H7, microcontrollori). Il progetto copre l'intero workflow: preparazione del dataset, addestramento con transfer learning su architetture fully-convolutional, quantizzazione INT8 con TFLite e benchmarking multi-dispositivo.

---

## Indice

1. [Installazione](#installazione)
2. [Fase 1 — Preparazione del Dataset](#fase-1--preparazione-del-dataset)
3. [Fase 2 — Analisi del Dataset](#fase-2--analisi-del-dataset)
4. [Fase 3 — Baseline con ML Classico](#fase-3--baseline-con-ml-classico)
5. [Fase 4 — Transfer Learning (Deep Learning)](#fase-4--transfer-learning-deep-learning)
6. [Fase 5 — Esportazione e Quantizzazione TFLite](#fase-5--esportazione-e-quantizzazione-tflite)
7. [Fase 6 — Benchmarking su Dispositivo](#fase-6--benchmarking-su-dispositivo)
8. [Risultati](#risultati)
9. [Prossimi Sviluppi](#prossimi-sviluppi)
10. [Ringraziamenti](#ringraziamenti)

---

## Installazione

```bash
# Clona il repository
git clone https://github.com/RealMattio/AI-mosquito-recognition-for-low-resources-devices.git
cd AI-mosquito-recognition-for-low-resources-devices

# Crea e attiva un ambiente virtuale (consigliato: Python 3.10)
conda create -n mosquito-env python=3.10
conda activate mosquito-env

# Installa le dipendenze
# Nota: networkx==3.5 richiede Python >=3.11; su 3.10 viene automaticamente
# installata l'ultima versione compatibile (3.4.x)
pip install -r requirements.txt
```

> **Dipendenze principali:** TensorFlow 2.19, Keras 3.10, PyTorch 2.7, scikit-learn 1.6, OpenCV 4.11, tensorflow-model-optimization 0.8.

---

## Fase 1 — Preparazione del Dataset

Il progetto usa il dataset **[Insects Recognition](https://www.kaggle.com/datasets/hammaadali/insects-recognition)** da Kaggle, che contiene 5 classi: *butterfly, dragonfly, grasshopper, ladybird, mosquito*. La pipeline lo trasforma in un problema di classificazione binaria (Mosquito / Not_Mosquito) e genera versioni augmentate per bilanciare e arricchire i dati.

### Passo 1 — Download

```bash
python download_dataset.py
```

Lo script usa `kagglehub` per scaricare il dataset e copiarlo in `./dataset/`. La struttura risultante è:

```
dataset/
├── Butterfly/
├── Dragonfly/
├── Grasshopper/
├── Ladybird/
└── Mosquito/
```

> **Prerequisito:** è necessario avere credenziali Kaggle configurate (`~/.kaggle/kaggle.json`).

### Passo 2 — Data Augmentation e Unificazione delle Classi

```bash
python data_augmentation.py
```

Lo script esegue due operazioni:
1. **Unificazione:** sposta tutte le immagini non-zanzara (butterfly, dragonfly, grasshopper, ladybird) in un'unica cartella `dataset/Not_Mosquito/`, rinominando i file con il prefisso della classe originale per evitare collisioni.
2. **Augmentation:** per ogni immagine genera 5 varianti con trasformazioni casuali (flip orizzontale, rotazione ±20%, zoom ±20%, variazione di contrasto ±20%, variazione di luminosità ±20%).

Output in `./augmented_dataset/`:
```
augmented_dataset/
├── Mosquito/          # Immagini originali + 5 augmentate per ciascuna
└── Not_Mosquito/      # Immagini originali + 5 augmentate per ciascuna
```

> Lo script chiede conferma interattiva prima di sovrascrivere una cartella già esistente.

### Passo 3 — Split Train / Validation / Test

```bash
python split_dataset.py
```

Divide `augmented_dataset/` in modo stratificato (70% train, 20% validation, 10% test) e crea la struttura:

```
augmented_dataset_splitted/
├── train/
│   ├── Mosquito/
│   └── Not_Mosquito/
├── validation/
│   ├── Mosquito/
│   └── Not_Mosquito/
└── test/
    ├── Mosquito/
    └── Not_Mosquito/
```

> La cartella di destinazione viene ricreata da zero ad ogni esecuzione.

---

## Fase 2 — Analisi del Dataset

```bash
python dataset_stats.py
```

Analizza la distribuzione delle immagini in una cartella del dataset (di default `augmented_dataset_splitted/test`). Stampa a schermo:
- Numero totale di classi e immagini
- Distribuzione per classe con percentuali
- Formati presenti (`.jpg`, `.png`, …)
- Dimensioni più comuni delle immagini

Genera anche un grafico a barre (`distribuzione_classi.png`) salvato nella stessa cartella analizzata.

Per analizzare una cartella diversa, modifica la variabile `percorso_cartella_dataset` in fondo al file oppure chiama la funzione `analizza_dataset()` direttamente:

```python
from dataset_stats import analizza_dataset
analizza_dataset('augmented_dataset_splitted/train')
```

---

## Fase 3 — Baseline con ML Classico

```bash
python ML_models_comparison.py
```

Addestra e valuta quattro modelli di ML classico (SVC lineare, MLP, RandomForest, AdaBoost) sulle immagini appiattite a 16.384 feature (128×128 pixel). Il workflow è:

1. Carica immagini da `./augmented_dataset/` e le ridimensiona a 128×128.
2. Codifica le etichette con `LabelEncoder`.
3. Split 80/20 con `train_test_split` stratificato.
4. Standardizza con `StandardScaler`.
5. Addestra i modelli e salva i file `.joblib` in `models/`.
6. Genera curve ROC e salva metriche in `metrics_results.csv` e report testuali in `reports/`.

> **Nota:** il blocco di training è commentato per default nel codice; attiva il training decommentando la sezione `model.fit()` e `joblib.dump()` all'interno del ciclo. Se i modelli sono già stati addestrati e salvati in `models/`, lo script carica direttamente quelli.

---

## Fase 4 — Transfer Learning (Deep Learning)

### Entry point

```bash
# Con pesi della base congelati (default — feature extraction pura)
python main.py

# Con pesi della base scongelati (fine-tuning completo)
python main.py --unfreeze
```

### Cosa fa `main.py`

Istanzia la classe `TransferLearning` (definita in `transfer_learning_tf.py`) con la seguente configurazione di default:

| Parametro | Valore |
|---|---|
| Image size | 96 × 96 |
| Epoche max | 100 |
| Early stopping patience | 10 epoche su `val_accuracy` |
| Learning rate | 0.001 (Adam) |
| K-fold cross-validation | 5 fold (StratifiedKFold) |
| ReduceLROnPlateau patience | 5 epoche su `val_loss`, factor 0.2 |
| Class weighting | automatico (`balanced`) |

### Architettura dei modelli (`noDense_model.py`)

Tutti i modelli pretrained usano una **testa fully-convolutional** al posto di layer Dense, per minimizzare le dimensioni del modello su microcontrollori:

```
Base pretrained (es. MobileNetV2, pesi ImageNet, congelata o no)
  └── GlobalAveragePooling2D
      └── Reshape (batch, 1, 1, C)
          └── Dropout(0.2)
              └── Conv2D 1×1 (num_classes, softmax)
                  └── Flatten → output
```

I modelli disponibili sono: `ResNet50`, `MobileNetV2`, `MobileNet`, `NASNetMobile`, `CustomCNN_Conv1D_Classifier`, `CustomCNN_Conv2D_Classifier`. Per selezionarne un sottoinsieme, decommenta e modifica le righe `models_names` e `name_to_save_models` in `main.py`.

### Output

```
keras_training/
├── keras_models_DDMM/
│   └── <nome_modello>_final_model.keras
└── keras_models_DDMM_performances_and_history/
    ├── history_<nome_modello>.json
    ├── evaluation_metrics_<nome_modello>.json
    ├── confusion_matrix_<nome_modello>.png
    ├── roc_curve_<nome_modello>.png
    └── final_training_results_DDMM.png
```

---

## Fase 5 — Esportazione e Quantizzazione TFLite

```bash
# Conversione base con quantizzazione INT8
python benchmark/keras_to_tflite.py --model MobileNetV2

# Con pruning (sparsità 50%) e fine-tuning prima della quantizzazione
python benchmark/keras_to_tflite.py --model MobileNetV2 --pruning --target_sparsity 0.5 --pruning_epochs 15 --early_stopping_patience 3
```

I modelli supportati dall'argomento `--model` sono: `MobileNetV2`, `ResNet50`, `NASNetMobile`, `CustomCNN`.

### Cosa fa lo script

1. Carica il file `.keras` corrispondente da `keras_training/keras_models_DDMM/`.
2. (Opzionale) Applica **weight pruning** con `tensorflow-model-optimization` e fa fine-tuning sul dataset di training.
3. Esegue **quantizzazione INT8 post-training** usando un dataset di calibrazione (100 campioni da `augmented_dataset_splitted/validation`).
4. Salva il file `.tflite` in `tflite_models/tflite_models_DDMM/`.

> **Nota:** i percorsi `KERAS_MODELS_DIR` e `TFLITE_MODELS_DIR` sono hardcoded con la data `1607` nel file. Aggiornali prima dell'esecuzione per puntare alla cartella del training corrente.

### Valutazione accuratezza TFLite

```bash
python benchmark/evaluate_tflite.py
```

Esegue il modello TFLite quantizzato sull'intero test set e restituisce accuracy, precision, recall e F1.

---

## Fase 6 — Benchmarking su Dispositivo

```bash
python benchmark/run_benchmark.py
```

Esegue **50 inferenze consecutive** su un'immagine di test (`benchmark/original_00000_original.png`) usando tutti i modelli `.keras` e `.tflite` trovati nelle cartelle configurate. Salva i risultati in un JSON strutturato con:
- Informazioni hardware e software del sistema (CPU, RAM, GPU via pynvml)
- Tempi medi, deviazione standard e lista completa dei singoli tempi per ogni modello

> **Configurazione:** modifica `KERAS_MODELS_DIR`, `TFLITE_MODELS_DIR` e `RESULTS_JSON_PATH` in cima al file prima dell'esecuzione. L'immagine di test deve essere copiata manualmente in `benchmark/`.

### Deployment su microcontrollore (Arduino Portenta H7)

```bash
# Preprocessing dell'immagine di test in formato binario per MicroPython
python benchmark/preprocess_image_to_bin.py

# Conversione PNG → BMP per compatibilità con OpenMV
python benchmark/convert_png_to_bmp.py
```

Gli script `benchmark/benchmark_portenta.py` e `benchmark/benchmark_portenta2.py` contengono il codice MicroPython da eseguire direttamente sull'Arduino Portenta H7 via OpenMV IDE.

---

## Risultati

I risultati completi sono riportati in [`benchmark/benchmark_results.md`](benchmark/benchmark_results.md).

| Modello | Formato | Dimensione (KB) | Accuratezza (%) | Inferenza PC (ms) | Inferenza RPi 5 (ms) |
|---|---|---|---|---|---|
| ResNet50 | keras | 92844 | 98.24 | 57.27 ± 4.38 | 235.00 ± 25.70 |
| ResNet50 | tflite int8 | 23679 | — | 69.11 ± 0.61 | 69.37 ± 0.34 |
| NASNetMobile | keras | 19393 | 96.78 | 94.28 ± 20.17 | 145.19 ± 30.29 |
| NASNetMobile | tflite int8 | 5226 | — | 68.27 ± 0.74 | 78.30 ± 1.31 |
| MobileNetV2 | keras | 9437 | 96.18 | 72.03 ± 3.14 | 120.76 ± 22.92 |
| MobileNetV2 | tflite int8 | 2560 | — | 7.22 ± 0.11 | 10.08 ± 0.11 |
| MobileNetV2 pruned | tflite int8 | 2540 | — | 7.26 ± 0.35 | 9.96 ± 0.15 |

> PC: AMD Ryzen 5 5600G, 56 GB RAM, RTX 3050 (WSL). RPi 5: Cortex-A76, 4 GB RAM.

La quantizzazione INT8 riduce la dimensione di MobileNetV2 da ~9 MB a ~2.5 MB con perdita trascurabile di accuratezza, e abbatte la latenza da 120 ms a ~10 ms su Raspberry Pi 5.

---

## Prossimi Sviluppi

Le analisi descritte in [`nuove_ricerche.md`](nuove_ricerche.md) rappresentano i prossimi step per portare il progetto a livello di pubblicazione scientifica:

### 1. Profilazione Energetica e Memory Footprint
Misurare la RAM occupata dai modelli prima e dopo la quantizzazione e stimare il consumo di potenza (Watt) della Raspberry Pi 5 durante inferenza continua vs. idle. Questo dato è fondamentale per validare la parola "sostenibilità" in contesti di edge computing per la Global Health.

### 2. Analisi di Robustezza ("In-the-Wild")
Valutare il modello TFLite su versioni perturbate sinteticamente del test set: *Gaussian Blur* (simula fuoco perso o mosso), riduzione della luminosità al 50% (condizioni serali), *Gaussian Noise* (rumore del sensore). Un grafico della caduta di F1-Score al crescere della perturbazione dimostra la robustezza del sistema in condizioni reali.

### 3. Threshold Tuning e Curve Precision-Recall
Variare la soglia decisionale da 0.1 a 0.9 per mostrare come sia possibile spingere la **Recall** del modello vicino al 99% abbassando la soglia (es. a 0.3), accettando un calo controllato della Precision. In epidemiologia un falso negativo (zanzara non rilevata) è più costoso di un falso positivo, e questa analisi dimostra la capacità di adattare il modello al requisito clinico.

### 4. Error Analysis — Analisi Qualitativa dei Fallimenti
Estrarre e visualizzare le immagini di Falsi Positivi e Falsi Negativi prodotti da MobileNetV2. Identificare pattern ricorrenti (es. insetti con morfologia simile alla zanzara come le tipule) permette di capire i limiti strutturali del modello e di discuterli criticamente nella tesi/paper.

---

## Ringraziamenti

* Kaggle community per il dataset **Insects Recognition**
* TensorFlow Lite team per la documentazione sulla ottimizzazione dei modelli
* I professori che mi seguono nel percorso di tesi

---

> *"The deadliest animal in the world is the mosquito."*
> — Bill Gates, 2014

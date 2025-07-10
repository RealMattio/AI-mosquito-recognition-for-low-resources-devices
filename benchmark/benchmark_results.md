# Benchmark di Inferenza per i Modelli Analizzati

## 🔧 Setup di Benchmark

Questo report riassume le performance di diversi modelli di classificazione su varie piattaforme hardware. Per l'esecuzione del benchmark sui vari dispositivi sono state utilizzate, ove possibile, le stesse versioni delle librerie.

* **Software:**
    * **Framework:** TensorFlow v2.19 / TFLite for Microcontrollers
    * **Librerie Principali:** Keras, NumPy
* **Hardware di Test:**
    * **Device 1:** Raspberry Pi 5 (OS: Raspberry Pi OS)
    * **Device 2:** Arduino Portenta H7 (Firmware: OpenMV/MicroPython)
    * **Device 3:** Arducam Pico4ML (Firmware: MicroPython)
    * **Device 4:** Arduino Nano 33 BLE Sense (Firmware: OpenMV/MicroPython)

---

## Risultati dei Benchmark
Di seguito sono riportate le performance dei device sui quali sono stati eseguiti i modelli in formato `.tflite` quantizzati a 8 bit e in formato keras sui device in grado di supportarli

1️⃣**Device**: *PC (Baseline)* - CPU: AMD Ryzen 5 5600G, RAM: 56 GB, System: WSL - Linux, GPU: NVIDIA GeForce RTX 3050

| Modello             | Formato | Dimensione (KB) | Dimensione Tensor Arena (KB)|Accuratezza del modello (%) | Tempo medio di inferenza (ms) |
| ------------------- | ------- | --------------- | --------------------------- | -------------------------- | ------------------------------|
| ResNet50            | keras   |     92844       |             ---             |          98.24             |       57.27 +- 4.38           |
| ResNet50            | tflite  |     23679       |             XX              |             XX             |       69.11 +- 0.61           |
| ResNet50 pruned     | tflite  |     23672       |             XX              |             XX             |       69.20 +- 0.91           |
| NASNetMobile        | keras   |     19393       |             ---             |          96.78             |       94.28 +- 20.17          |
| NASNetMobile        | tflite  |      5226       |             XX              |             XX             |       68.27 +- 0.74           |
| NASNetMobile pruned | tflite  |      5192       |             XX              |             XX             |       67.53 +- 0.55           |
| MobileNetV2         | keras   |      9437       |             ---             |          96.18             |       72.03 +- 3.14           |
| MobileNetV2         | tflite  |      2560       |             XX              |             XX             |       7.22 +- 0.11            |
| MobileNetV2 pruned  | tflite  |      2540       |             XX              |             XX             |       7.26 +- 0.35            |
| CustomCNN           | XX      |        XX       |             XX              |             XX             |            XX                 |

2️⃣**Device**: Raspberry Pi 5 - CPU: Cortex-A76, RAM: 4 GB, System: Linux, GPU: None

| Modello             | Formato | Dimensione (KB) | Dimensione Tensor Arena (KB)|Accuratezza del modello (%) | Tempo medio di inferenza (ms) |
| ------------------- | ------- | --------------- | --------------------------- | -------------------------- | ------------------------------|
| ResNet50            | keras   |     92844       |             ---             |          98.24             |      235.00 +- 25.70          |
| ResNet50            | tflite  |     23679       |             XX              |             XX             |       69.37 +- 0.34           |
| ResNet50 pruned     | tflite  |     23672       |             XX              |             XX             |       67.54 +- 1.27           |
| NASNetMobile        | keras   |     19393       |             ---             |          96.78             |      145.19 +- 30.29          |
| NASNetMobile        | tflite  |      5226       |             XX              |             XX             |       78.30 +- 1.31           |
| NASNetMobile pruned | tflite  |      5192       |             XX              |             XX             |       78.17 +- 1.31           |
| MobileNetV2         | keras   |      9437       |             ---             |          96.18             |      120.76 +- 22.92          |
| MobileNetV2         | tflite  |      2560       |             XX              |             XX             |       10.08 +- 0.11           |
| MobileNetV2 pruned  | tflite  |      2540       |             XX              |             XX             |        9.96 +- 0.15           |
| CustomCNN           | XX      |        XX       |             XX              |             XX             |            XX                 |

3️⃣**Device**: Arduino Portenta H7 - INSERIRE LE SPECIFICHE TECNICHE

| Modello             | Formato | Dimensione (KB) | Dimensione Tensor Arena (KB)|Accuratezza del modello (%) | Tempo medio di inferenza (ms) |
| ------------------- | ------- | --------------- | --------------------------- | -------------------------- | ------------------------------|
| ResNet50            | tflite  |     23679       |             XX              |             XX             |            XX                 |
| ResNet50 pruned     | tflite  |     23672       |             XX              |             XX             |            XX                 |
| NASNetMobile        | tflite  |      5226       |             XX              |             XX             |            XX                 |
| NASNetMobile pruned | tflite  |      5192       |             XX              |             XX             |            XX                 |
| MobileNetV2         | tflite  |      2560       |             XX              |             XX             |            XX                 |
| MobileNetV2 pruned  | tflite  |      2540       |             XX              |             XX             |            XX                 |
| CustomCNN           | XX      |        XX       |             XX              |             XX             |            XX                 |



## 📒Note e Osservazioni

COMPLETARE LE NOTE E LE OSSERVAZIONI

* La quantizzazione INT8 riduce drasticamente la dimensione dei modelli (da ~98MB a ~6.5MB per ResNet50) con una perdita di accuratezza minima (<1%).
* Il modello `CustomCNN` è il più leggero e l'unico eseguibile su tutti i dispositivi, incluso l'Arduino Nano 33.
* Le architetture più complesse come ResNet50 non sono eseguibili sui microcontrollori con meno memoria (Pico4ML, Nano 33) a causa delle dimensioni dell'arena dei tensori.
* Il Raspberry Pi 5 e il Portenta H7 mostrano le performance migliori in termini di latenza di inferenza (dati da report separato).
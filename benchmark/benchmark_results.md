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
Di seguito sono riportate le performance dei device sui quali sono stati eseguiti i modelli in formato `.tflite`. 

1️⃣**Device**: *PC (Baseline)* - INSERIRE LE SPECIFICHE TECNICHE

| Modello      | Dimensione (KB) | Dimensione Tensor Arena (KB)|Accuratezza del modello (%) | Tempo medio di inferenza (ms) |
| ------------ | --------------- | --------------------------- | -------------------------- | ------------------------------|
| ResNet50     |        XX       |             XX              |             XX             |            XX                 |
| NASNetMobile |        XX       |             XX              |             XX             |            XX                 |
| MobileNetV2  |        XX       |             XX              |             XX             |            XX                 |
| CustomCNN    |        XX       |             XX              |             XX             |            XX                 |

2️⃣**Device**: Raspberry Pi 5 - INSERIRE LE SPECIFICHE TECNICHE

| Modello      | Dimensione (KB) | Dimensione Tensor Arena (KB)|Accuratezza del modello (%) | Tempo medio di inferenza (ms) |
| ------------ | --------------- | --------------------------- | -------------------------- | ------------------------------|
| ResNet50     |        XX       |             XX              |             XX             |            XX                 |
| NASNetMobile |        XX       |             XX              |             XX             |            XX                 |
| MobileNetV2  |        XX       |             XX              |             XX             |            XX                 |
| CustomCNN    |        XX       |             XX              |             XX             |            XX                 |

3️⃣**Device**: Raspberry Pi 5 - INSERIRE LE SPECIFICHE TECNICHE

| Modello      | Dimensione (KB) | Dimensione Tensor Arena (KB)|Accuratezza del modello (%) | Tempo medio di inferenza (ms) |
| ------------ | --------------- | --------------------------- | -------------------------- | ------------------------------|
| ResNet50     |        XX       |             XX              |             XX             |            XX                 |
| NASNetMobile |        XX       |             XX              |             XX             |            XX                 |
| MobileNetV2  |        XX       |             XX              |             XX             |            XX                 |
| CustomCNN    |        XX       |             XX              |             XX             |            XX                 |



## 📒Note e Osservazioni

COMPLETARE LE NOTE E LE OSSERVAZIONI

* La quantizzazione INT8 riduce drasticamente la dimensione dei modelli (da ~98MB a ~6.5MB per ResNet50) con una perdita di accuratezza minima (<1%).
* Il modello `CustomCNN` è il più leggero e l'unico eseguibile su tutti i dispositivi, incluso l'Arduino Nano 33.
* Le architetture più complesse come ResNet50 non sono eseguibili sui microcontrollori con meno memoria (Pico4ML, Nano 33) a causa delle dimensioni dell'arena dei tensori.
* Il Raspberry Pi 5 e il Portenta H7 mostrano le performance migliori in termini di latenza di inferenza (dati da report separato).
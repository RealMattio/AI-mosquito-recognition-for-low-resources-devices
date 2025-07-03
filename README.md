# AI-mosquito-recognition-for-low-resources-devices
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.12-blue?logo=python)
![Static Badge](https://img.shields.io/badge/MicroPython-black?style=flat&logo=micropython&logoColor=white)
![Static Badge](https://img.shields.io/badge/Visual_Studio_Code-0078D4?style=flat&logo=visual%20studio%20code&logoColor=white)
![Static Badge](https://img.shields.io/badge/Arduino_IDE-00979D?style=flat&logoColor=white&logo=arduino)



![Static Badge](https://img.shields.io/badge/Mosquito🦟-friends-green)
![GitHub commit activity](https://img.shields.io/github/commit-activity/m/RealMattio/AI-mosquito-recognition-for-low-resources-devices?style=flat&label=Commit%20activity&color=yellow)


<!-- [![Build](https://img.shields.io/badge/build-passing-brightgreen?logo=githubactions)](#) -->
<!--[![License](https://img.shields.io/badge/license-MIT-blue)](#license)-->

A **TinyML / Computer-Vision** workflow that **detects mosquitoes** on
resource-constrained devices (e.g. Raspberry Pi Pico, Arduino
Portenta).  
The repository benchmarks several lightweight CNN and classic
machine-learning pipelines, exports the best models to **TensorFlow Lite** and
provides end-to-end scripts for **training, evaluation and on-device
inference**.

---

## ✨ Features
| Area | What you get |
|------|--------------|
| **Dataset tooling** | Class-merging, split & data-augmentation |
| **Model zoo** | • Classical ML baselines (SVC, RandomForest, MLP, AdaBoost) <br>• Pre-trained MobileNet V2, ResNet50, NASNetMobile transfer learning <br>|
| **Export & quantisation** | Float → int8 post-training quantisation, TFLite models ready for MCU / EdgeTPU |
| **Benchmark suite** | Accuracy, model-size, inference-time comparison on microcontroller, PC  **and** Raspberry Pi 5|

---

## 📂 Repository layout

```
.
├── ML_DL_training_compare/    #  Graph, reports, results about some models
├── keras_training/            #  Keras models and their performances 
├── benchmark/                 #  Deploying on microcontrollers for time benchmarking
├── tflite_models/             #  Models converted in tflite extension
├── data_augmentation.py       #  Scripts for data augmentation
├── download_dataset.py        #  Scripts for downloading dataset into ./dataset folder
├── images_preprocessing.py    #  Class for preprocessing: resize, rescale, etc.
├── split_dataset.py           #  Stratified train/val/test split in folders
├── ML_models_comparison.py    #  Scriptf for comparing some ML models 
├── transfer_learning.py       #  Transfer learning with PyTorch
├── transfer_learning_tf.py    #  Transfer learning with TensorFlow-Keras
└── main.py                    #  Entry-point

````

---

## 🗄️ Dataset

To train the models covered in this project we rely on the open **Insects Recognition** image set from Kaggle, downloadable at this [link](https://www.kaggle.com/datasets/hammaadali/insects-recognition?resource=download).
The original dataset provide 5 diffrent class of insects: *butterfly, dragonfly, grasshopper, ladybird,* and *mosquito*. In this project the aim is to classify **mosquitos** so the other classes was unified in one class called **non_mosquitos**.
To download the dataset run script `download_dataset.py`, than to prepare the data for training run `data_augmentation.py` and than `split_dataset.py`.

> **Remember:** the download script will grab and unpack the images automatically, but the scripts just mentioned must be run in the correct order.

---

## 🚀 Quick-start

```bash
# 1. Clone
git clone https://github.com/RealMattio/AI-mosquito-recognition-for-low-resources-devices.git
cd AI-mosquito-recognition-for-low-resources-devices
```
```bash
# 2. Set up Python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt         # TensorFlow 2.x, scikit-learn, OpenCV 
```
<!--
> **💡 Headless Raspberry Pi?**
> Replace `--camera 0` with `--image path/to/photo.jpg` or
> `--folder path/to/folder` to batch-process images offline.
-->
---

## 📊 Results

| Model | Params    | Size (MB) | Acc. (% test) | Inference time (ms) |
| ------| --------- | --------- | ------------- | ---------- |
| MobileNet V2 (fine-tuned) | 3.5 M     | 14    | N/A  | N/A |
| ResNet50 (fine-tuned)     | 25.6 M    | 98    | N/A  | N/A |
| NASNetMobile (fine-tuned) | 5.3 M     | 23    | N/A  | N/A |


> The table reports keras files, Python 3.10.
> **This table may be not updated.**

<!--
> The table reports int8-quantised TFLite files evaluated on a Raspberry Pi 4
> B (4 GB) running 64-bit Raspberry Pi OS bookworm, Python 3.10.
> **Update the numbers after re-training on your hardware.**
-->
---

## 🛠️ Deployment guides

More information will be made available soon

---

## 🤝 Contributing

More information will be made available soon
<!--
1. Fork the project & create a feature branch
2. Follow [`black`](https://black.readthedocs.io/) + `ruff` for code style
3. Add unit tests in `tests/` (*pytest*)
4. Open a pull request – automatic benchmarks will run on each PR

Please open an issue first if you plan to work on major changes (new model
architectures, audio classification, etc.).
-->
---

## 🗺️ Roadmap

* [ ] Audio-based mosquito detection (HumBugDB)
* [ ] YOLO-Nano object detection for tracking mosquitoes in video
* [ ] Docker + Compose stack for edge-to-cloud telemetry
* [ ] Automatic OTA model updates for deployed devices

---
<!--
## 📄 License

---
-->

## 🙏 Acknowledgements

* Kaggle community for the **Insects Recognition** dataset
* TensorFlow Lite team for excellent documentation on model optimisation
* All contributors to open-source TinyML tooling and the wider entomology
  research community
* My professors who follow me day by day in my thesis path
---

> “The deadliest animal in the world is the mosquito.”
> *— Bill Gates, 2014*
import torch, onnx
#from onnx2tf.backend import prepare
import tensorflow as tf
import tf2onnx
from torchvision import models
import torch.nn as nn


# Path del modello PyTorch salvato da convertire
SOURCE_PATH = "saved_models\ResNet18_0.8659.pth"
DESTINATION_PATH = "saved_models\ResNet18_0.8659"
DESTINATION_ONNX_PATH = DESTINATION_PATH + ".onnx"
DESTINATION_TF_PATH = DESTINATION_PATH + "_tf"
DESTINATION_TFLITE_PATH = DESTINATION_PATH + ".tflite"
""" 
# 1. Carica modello PyTorch
print(f"Caricamento modello PyTorch da {SOURCE_PATH}...")
model = torch.load(SOURCE_PATH, map_location="cpu")
model.eval()
 """

# 2. Carica il state_dict e deduci il numero di classi
state_dict = torch.load(SOURCE_PATH, map_location="cpu")
# assumes state_dict keys include "fc.weight"
num_classes = state_dict["fc.weight"].size(0)
print(f"Numero di classi rilevato: {num_classes}")

# 3. Instanzia ResNet18 e sostituisci l'ultimo layer
model = models.resnet18(pretrained=False)
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

# 4. Carica i pesi e metti in eval mode
model.load_state_dict(state_dict)
model.eval()

# 2. Esporta in ONNX
print(f"Esportazione modello in ONNX a {DESTINATION_ONNX_PATH}...")
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    model, dummy, DESTINATION_ONNX_PATH,
    input_names=['input'], output_names=['output'],
    opset_version=11
)

# 3. ONNX → TensorFlow SavedModel
print(f"Conversione modello ONNX a TensorFlow SavedModel...")
onnx_model = onnx.load(DESTINATION_ONNX_PATH)
onnx.checker.check_model(onnx_model)

spec = (tf.TensorSpec((None, 3, 224, 224), tf.float32, name="input"),)
tf_rep = tf2onnx.convert.from_onnx(onnx_model, input_signature=spec, output_path=DESTINATION_TF_PATH)


# 4. SavedModel → TFLite
print(f"Conversione modello TensorFlow SavedModel a TFLite a {DESTINATION_TFLITE_PATH}...")
converter = tf.lite.TFLiteConverter.from_saved_model(DESTINATION_TF_PATH)
# (opzionale) ottimizzazione quant
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
with open(DESTINATION_TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

print(f"✅ {DESTINATION_TFLITE_PATH} creato")

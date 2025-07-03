# Script da eseguire sul tuo PC, non sul Portenta
import tensorflow as tf

TFLITE_MODEL_PATH = 'tflite_models/tflite_models_2506/MobileNetV2_quant_int8.tflite'

try:
    # Carica il modello TFLite
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    interpreter.allocate_tensors() # Questo è il comando che fallisce sul Portenta

    print("SUCCESSO: Il modello è valido e si carica correttamente su PC.")
    print("\nEcco i dettagli del modello:")
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    print(f"  Input Shape: {input_details['shape']}")
    print(f"  Input Type: {input_details['dtype']}")
    print(f"  Output Shape: {output_details['shape']}")
    print(f"  Output Type: {output_details['dtype']}")

except Exception as e:
    print(f"ERRORE: Il file del modello sembra essere corrotto o non valido.\n{e}")
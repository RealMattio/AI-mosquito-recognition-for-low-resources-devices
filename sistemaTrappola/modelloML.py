import tensorflow as tf
import numpy as np
from typing import TypeAlias, List
from interfacce import IPredizione 

# Definiamo un tipo personalizzato per l'immagine. 
# Per questa implementazione, assumiamo sia un array NumPy.
Immagine: TypeAlias = np.ndarray

class ModelloML(IPredizione):
    """
    Classe che implementa l'interfaccia IPredizione.
    Rappresenta un modello di machine learning TFLite che effettua predizioni
    su immagini.
    """

    def __init__(self, model_path: str, class_names: List[str]):
        """
        Costruttore della classe. Carica il modello TFLite e prepara l'interprete.

        Args:
            model_path (str): Il percorso al file del modello .tflite.
            class_names (List[str]): Una lista di stringhe con i nomi delle classi,
                                     nell'ordine corretto (es. ['Not_Mosquito', 'Mosquito']).
        """
        self.stato: str = "Caricamento..."
        self.class_names = class_names
        
        try:
            # Carica l'interprete TFLite e alloca i tensori
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()

            # Ottieni i dettagli di input e output per un uso futuro
            self.input_details = self.interpreter.get_input_details()[0]
            self.output_details = self.interpreter.get_output_details()[0]
            
            self.stato = "Pronto"
            print(f"Modello ML caricato da '{model_path}'. Stato: {self.stato}.")
            print(f"  - Input richiesto: {self.input_details['shape']}, Tipo: {self.input_details['dtype'].__name__}")
            print(f"  - Output restituito: {self.output_details['shape']}, Tipo: {self.output_details['dtype'].__name__}")

        except Exception as e:
            self.stato = "Errore"
            print(f"ERRORE: Impossibile caricare il modello TFLite da '{model_path}'.")
            print(f"Dettagli: {e}")

    def effettuaPredizione(self, img: Immagine) -> str:
        """
        Esegue l'inferenza su una singola immagine.

        Args:
            img (Immagine): L'immagine da analizzare, come array NumPy (H, W, C).

        Returns:
            str: Il nome della classe predetta (es. "Mosquito").
        """
        if self.stato != "Pronto":
            return "Modello non pronto"

        # --- 1. Preprocessing dell'Immagine ---
        # Ottieni le dimensioni di input richieste dal modello
        _, height, width, _ = self.input_details['shape']
        
        # Ridimensiona l'immagine e aggiungi la dimensione del batch
        img_resized = tf.image.resize(img, [height, width])
        input_data = np.expand_dims(img_resized, axis=0)

        # Converte il tipo di dati e quantizza se necessario (per modelli INT8)
        if self.input_details['dtype'] == np.int8:
            # Formula di quantizzazione standard
            input_scale, input_zero_point = self.input_details['quantization']
            input_data = (input_data / input_scale + input_zero_point).astype(np.int8)
        
        # --- 2. Esecuzione dell'Inferenza ---
        # Imposta il tensore di input e invoca l'interprete
        self.interpreter.set_tensor(self.input_details['index'], input_data)
        self.interpreter.invoke()

        # --- 3. Post-processing del Risultato ---
        # Ottieni il tensore di output (contiene le probabilità o i logits)
        output_data = self.interpreter.get_tensor(self.output_details['index'])
        
        # Trova l'indice della classe con il valore più alto
        predicted_index = np.argmax(output_data[0])
        
        # Restituisci il nome della classe corrispondente
        predicted_class_name = self.class_names[predicted_index]
        
        return predicted_class_name
    

'''
# --- TEST ---
if __name__ == '__main__':
    # Questo blocco verrà eseguito solo se esegui direttamente questo file.
    
    # 1. Definisci i parametri
    # Assicurati che il percorso del modello e i nomi delle classi siano corretti
    PATH_MODELLO_TFLITE = 'tflite_models/tflite_models_2506/MobileNetV2_quant_int8.tflite'
    NOMI_CLASSI = ['Not_Mosquito', 'Mosquito']

    # 2. Crea un'immagine di test finta (un array NumPy casuale)
    # Nella realtà, questa immagine arriverebbe dalla fotocamera.
    # Dimensioni originali dell'immagine: 640x480 con 3 canali (RGB)
    immagine_test = np.random.randint(0, 256, size=(480, 640, 3), dtype=np.uint8)
    print(f"Creata immagine di test finta con dimensioni {immagine_test.shape}")

    # 3. Istanzia il modello
    modello = ModelloML(model_path=PATH_MODELLO_TFLITE, class_names=NOMI_CLASSI)

    # 4. Esegui la predizione se il modello è stato caricato correttamente
    if modello.stato == "Pronto":
        print("\nEsecuzione della predizione...")
        risultato = modello.effettuaPredizione(immagine_test)
        print(f"\n✅ Risultato della Predizione: '{risultato}'")
'''
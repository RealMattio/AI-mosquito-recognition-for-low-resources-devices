import numpy as np
from typing import TypeAlias, Tuple
from interfacce import ICamera

# Definiamo un tipo personalizzato per l'immagine. 
# Un array NumPy è la rappresentazione più comune e utile.
Immagine: TypeAlias = np.ndarray

class ModuloCAM(ICamera):
    """
    Classe che implementa l'interfaccia ICamera.
    Simula un modulo camera che può scattare foto. In questa versione,
    le foto sono generate casualmente a scopo di test.
    """
    
    def __init__(self, risoluzione: Tuple[int, int] = (640, 480)):
        """
        Costruttore della classe ModuloCAM.

        Args:
            risoluzione (Tuple[int, int]): La risoluzione (larghezza, altezza) 
                                           delle immagini da generare.
        """
        self.stato: str = "Pronto"
        self.risoluzione: Tuple[int, int] = risoluzione
        
        # In una implementazione reale, qui andrebbe il codice per
        # inizializzare la camera fisica (es. con picamera2 o opencv).
        # Esempio:
        # self.camera = picamera2.Picamera2()
        # config = self.camera.create_preview_configuration(main={"size": self.risoluzione})
        # self.camera.configure(config)
        # self.camera.start()
        
        print(f"ModuloCAM inizializzato (virtualmente) con risoluzione {self.risoluzione}. Stato: {self.stato}.")

    def scattaFoto(self) -> Immagine:
        """
        Simula lo scatto di una foto generando un'immagine casuale.

        Returns:
            Immagine: Un array NumPy che rappresenta l'immagine scattata.
        """
        print("📸 Scatto di una foto (simulato)...")
        
        # In una implementazione reale, qui si catturerebbe l'immagine dalla camera.
        # Esempio:
        # immagine_reale = self.camera.capture_array()
        # return immagine_reale
        
        # Per ora, generiamo un'immagine casuale con la risoluzione specificata.
        # La forma di un'immagine è (altezza, larghezza, canali_colore).
        height, width = self.risoluzione[1], self.risoluzione[0]
        
        # Genera un array di byte casuali (da 0 a 255) per rappresentare i pixel
        immagine_casuale = np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)
        
        print("Immagine casuale generata.")
        return immagine_casuale

'''
import matplotlib.pyplot as plt
# --- Esempio di Utilizzo ---
if __name__ == '__main__':
    # Questo blocco di codice verrà eseguito solo se esegui direttamente questo file.
    
    print("--- Inizio test del ModuloCAM ---")
    
    # 1. Crea un'istanza della camera con una risoluzione specifica
    camera_test = ModuloCAM(risoluzione=(1280, 720))
    
    # 2. Chiama il metodo per scattare una foto
    foto_scattata = camera_test.scattaFoto()
    
    # 3. Verifica le proprietà dell'immagine generata
    print("\n--- Analisi dell'immagine ricevuta ---")
    print(f"Tipo di oggetto restituito: {type(foto_scattata)}")
    if isinstance(foto_scattata, np.ndarray):
        print(f"Dimensioni dell'immagine (H, W, C): {foto_scattata.shape}")
        print(f"Tipo di dati dei pixel: {foto_scattata.dtype}")

    # --- Visualizzazione dell'immagine generata ---
    print("\nVisualizzazione dell'immagine generata...")
    plt.imshow(foto_scattata)
    plt.title("Immagine Casuale Generata")
    plt.axis('off')  # Nasconde gli assi per una visualizzazione più pulita
    plt.show() # Apre una finestra per mostrare l'immagine
    # ----------------------------------------------------


    print("\n--- Test terminato ---")

    '''
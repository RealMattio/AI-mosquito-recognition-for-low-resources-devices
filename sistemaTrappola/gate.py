from gpiozero import Servo
from time import sleep

class GATE:
    """
    Classe che rappresenta un gate (porta) fisico controllato da un servomotore.
    Gestisce le operazioni di apertura e chiusura.
    """

    def __init__(self, pin: int):
        """
        Costruttore della classe GATE.

        Args:
            pin (int): Il numero del pin GPIO (BCM) a cui è collegato 
                       il filo del segnale del servomotore.
        """
        self.pin: int = pin
        self.stato: str = "chiuso"  # Lo stato iniziale di default

        # --- Oggetto GPIO ---
        # Crea l'oggetto gpiozero che controlla il servomotore.
        # L'underscore indica che è una variabile "interna" alla classe.
        try:
            self._servo_gpio = Servo(self.pin)
            # All'avvio, ci assicuriamo che il servo sia in posizione di chiusura
            self._servo_gpio.min()
            print(f"Gate inizializzato sul pin GPIO {self.pin}. Stato: {self.stato}.")
        except Exception as e:
            self._servo_gpio = None
            print(f"ATTENZIONE: Impossibile inizializzare il GPIO sul pin {self.pin}. Il codice funzionerà in modalità 'virtuale'.")
            print(f"Dettagli errore: {e}")

    def apri(self):
        """
        Apre il gate muovendo il servomotore alla sua posizione massima.
        """
        if self.stato == "chiuso":
            if self._servo_gpio:
                self._servo_gpio.max() # Muove il servo a +90 gradi (o al suo massimo)
                sleep(1) # Lascia al servo il tempo di completare il movimento
            self.stato = "aperto"
            print(f"✅ Gate sul pin {self.pin} è stato aperto.")
        else:
            print(f"Gate sul pin {self.pin} è già aperto.")
            
    def chiudi(self):
        """
        Chiude il gate muovendo il servomotore alla sua posizione minima.
        """
        if self.stato == "aperto":
            if self._servo_gpio:
                self._servo_gpio.min() # Muove il servo a -90 gradi (o al suo minimo)
                sleep(1) # Lascia al servo il tempo di completare il movimento
            self.stato = "chiuso"
            print(f"❌ Gate sul pin {self.pin} è stato chiuso.")
        else:
            print(f"Gate sul pin {self.pin} è già chiuso.")

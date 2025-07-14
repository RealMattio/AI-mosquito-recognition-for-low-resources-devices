# Importiamo la classe LED dalla libreria gpiozero, rinominandola per evitare conflitti
from gpiozero import LED as GPIOLed

class LED:
    """
    Classe che rappresenta un singolo LED collegato a un pin GPIO di un Raspberry Pi.
    Gestisce lo stato e le operazioni di accensione e spegnimento.
    """

    def __init__(self, pin: int):
        """
        Costruttore della classe LED.

        Args:
            pin (int): Il numero del pin GPIO (BCM) a cui il LED è collegato.
        """
        # --- Attributi ---
        self.pin: int = pin
        self.stato: str = "spento"  # Stato iniziale del LED

        # --- Oggetto GPIO ---
        # Crea l'oggetto gpiozero che controlla fisicamente il pin.
        # L'underscore indica che è una variabile "interna" alla classe.
        self._led_gpio = GPIOLed(self.pin)
        
        print(f"LED inizializzato sul pin GPIO {self.pin}. Stato: {self.stato}.")

    def accendi(self):
        """
        Accende il LED e aggiorna lo stato.
        """
        if self.stato == "spento":
            self._led_gpio.on()
            self.stato = "acceso"
            print(f"💡 LED sul pin {self.pin} è stato acceso.")
        else:
            print(f"LED sul pin {self.pin} è già acceso.")

    def spegni(self):
        """
        Spegne il LED e aggiorna lo stato.
        """
        if self.stato == "acceso":
            self._led_gpio.off()
            self.stato = "spento"
            print(f"⚫ LED sul pin {self.pin} è stato spento.")
        else:
            print(f"LED sul pin {self.pin} è già spento.")

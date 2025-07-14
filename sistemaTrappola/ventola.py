from gpiozero import PWMLED

class VENTOLA:
    """
    Classe che rappresenta una ventola controllabile collegata a un pin GPIO
    di un Raspberry Pi. La sua velocità viene regolata tramite PWM.
    """

    def __init__(self, pin: int, stato_iniziale: str = "spenta", potenza_iniziale: int = 0):
        """
        Costruttore della classe VENTOLA.

        Args:
            pin (int): Il numero del pin GPIO (BCM) a cui la ventola è collegata.
            stato_iniziale (str): Lo stato di partenza ("accesa" o "spenta").
            potenza_iniziale (int): La potenza iniziale in percentuale (da 0 a 100).
        """
        self.pin: int = pin
        self.stato: str = stato_iniziale
        self.potenza: int = potenza_iniziale

        # Crea l'oggetto gpiozero che controlla fisicamente il pin con PWM.
        self._fan_gpio = PWMLED(self.pin)

        # Imposta la potenza iniziale al momento della creazione
        self.regola_potenza(self.potenza)
        
        print(f"Ventola inizializzata sul pin GPIO {self.pin}. Potenza: {self.potenza}%.")

    def regola_potenza(self, nuova_potenza: int):
        """
        Regola la velocità della ventola impostando la potenza in percentuale.

        Args:
            nuova_potenza (int): La nuova potenza desiderata, da 0 (spenta) a 100 (massimo).
        """
        # --- Validazione dell'Input ---
        # Assicuriamoci che il valore sia nel range corretto [0, 100]
        if nuova_potenza < 0:
            nuova_potenza = 0
            print("Attenzione: la potenza non può essere negativa. Impostata a 0.")
        elif nuova_potenza > 100:
            nuova_potenza = 100
            print("Attenzione: la potenza non può superare 100. Impostata al massimo.")

        # Aggiorna la potenza e lo stato interni
        self.potenza = nuova_potenza
        self.stato = "accesa" if self.potenza > 0 else "spenta"

        # --- Controllo Hardware ---
        # Converte la potenza (0-100) in un valore per il duty cycle PWM (0.0-1.0)
        duty_cycle = self.potenza / 100.0
        self._fan_gpio.value = duty_cycle

        print(f"🌀 Ventola sul pin {self.pin} impostata al {self.potenza}% di potenza.")

from observerPattern import IObserver
from typing import List
from interfacce import ICamera, IPredizione
from led import LED
from gate import GATE
from ventola import VENTOLA

class ControlloreTrappola(IObserver):
    """
    L'Osservatore (Observer). Viene notificato dal Sensore
    e agisce di conseguenza, orchestrando gli altri componenti.
    """
    def __init__(self,
                 lista_cam: List[ICamera],
                 lista_modelli_ml: List[IPredizione],
                 lista_led: List[LED],
                 lista_gate: List[GATE],
                 lista_ventole: List[VENTOLA]):
        
        self._stato: str = "Inattivo" # Privato per convenzione
        
        # Dipendenze e componenti controllati
        self.lista_cam = lista_cam
        self.lista_modelli_ml = lista_modelli_ml
        self.lista_led = lista_led
        self.lista_gate = lista_gate
        self.lista_ventole = lista_ventole

    def aggiorna(self):
        """
        Metodo richiesto dall'interfaccia IObserver.
        Viene chiamato dal SensoreMovimento quando rileva qualcosa.
        """
        # Qui verrebbe chiamata la logica di gestisciRilevamento()
        pass
        
    def gestisciRilevamento(self):
        """Metodo principale per gestire un rilevamento."""
        pass

    def _apriGate(self):
        pass

    def _chiudiGate(self):
        pass

    def _accendiLED(self):
        pass
    
    def _spegniLED(self):
        pass

    def _potenziaFAN(self, potenza: int):
        pass
        
    def _depotenziaFAN(self, potenza: int):
        pass
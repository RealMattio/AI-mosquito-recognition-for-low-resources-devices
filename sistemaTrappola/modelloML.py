from interfacce import IPredizione
from typing import TypeAlias

# Definiamo un tipo personalizzato per chiarezza
Immagine: TypeAlias = object

class ModelloML(IPredizione):
    """
    Classe che implementa l'interfaccia IPredizione.
    Rappresenta un modello di machine learning che effettua predizioni
    su immagini.
    """
    def __init__(self):
        self.stato: str = "Pronto"
        
    def effettuaPredizione(self, img: Immagine) -> str:
        pass
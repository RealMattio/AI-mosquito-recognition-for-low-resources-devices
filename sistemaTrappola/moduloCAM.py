from interfacce import ICamera
from typing import TypeAlias

# Definiamo un tipo personalizzato per chiarezza
Immagine: TypeAlias = object

class ModuloCAM(ICamera):
    """
    Classe che implementa l'interfaccia ICamera.
    Rappresenta un modulo di camera che può scattare foto.
    """
    def __init__(self):
        self.stato: str = "Inattivo"
        
    def scattaFoto(self) -> Immagine:
        pass
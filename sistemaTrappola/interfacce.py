from abc import ABC, abstractmethod
from typing import TypeAlias, List

# Definiamo un tipo personalizzato per chiarezza
Immagine: TypeAlias = object
# Interfacce per i servizi
class ICamera(ABC):
    """
    Interfaccia per il Modulo di Camera.
    Deve implementare il metodo 'scattaFoto' che restituisce
    un'immagine (di tipo Immagine).
    """
    @abstractmethod
    def scattaFoto(self) -> Immagine:
        pass

class IPredizione(ABC):
    """
    Interfaccia per il Modello di Machine Learning.
    Deve implementare il metodo 'effettuaPredizione' che accetta
    un'immagine e restituisce una stringa con la predizione.
    """
    @abstractmethod
    def effettuaPredizione(self, img: Immagine) -> str:
        pass

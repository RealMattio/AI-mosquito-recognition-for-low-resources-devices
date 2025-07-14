from observerPattern import IObserver, ISubject
from typing import List

class SensoreMovimento(ISubject):
    """
    Il Soggetto (Subject). Il suo compito è monitorare un evento
    e notificare chiunque sia interessato (gli Observer).
    """
    def __init__(self):
        self.stato: str = "In ascolto"
        self._observers: List[IObserver] = [] # Privato per convenzione
        
    def iscrivi(self, observer: IObserver):
        # Logica per aggiungere un observer alla lista
        pass
        
    def disiscrivi(self, observer: IObserver):
        # Logica per rimuovere un observer dalla lista
        pass
        
    def notifica(self):
        # Logica per notificare tutti gli observer
        pass
        
    def _controllaMovimento(self):
        """
        Metodo privato che, quando rileva un movimento,
        chiamerebbe self.notifica().
        """
        pass
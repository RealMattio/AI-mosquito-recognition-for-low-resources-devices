from abc import ABC, abstractmethod
# Interfacce per l'Observer Pattern
class IObserver(ABC):
    """
    Interfaccia per l'Osservatore (Observer) dell'Observer Pattern.
    Deve implementare il metodo 'aggiorna' che viene chiamato
    dal Soggetto (Subject) quando c'è un cambiamento di stato.
    """
    @abstractmethod
    def aggiorna(self):
        """Metodo chiamato dal Subject per notificare un cambiamento."""
        pass

class ISubject(ABC):
    """
    Interfaccia per il Soggetto (Subject) dell'Observer Pattern.
    Gestisce la lista degli observer e le operazioni di iscrizione,
    disiscrizione e notifica.
    """
    @abstractmethod
    def iscrivi(self, observer: IObserver):
        """Aggiunge un observer alla lista."""
        pass

    @abstractmethod
    def disiscrivi(self, observer: IObserver):
        """Rimuove un observer dalla lista."""
        pass

    @abstractmethod
    def notifica(self):
        """Notifica a tutti gli observer iscritti."""
        pass

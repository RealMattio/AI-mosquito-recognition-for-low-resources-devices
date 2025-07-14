from observerPattern import IObserver, ISubject
from typing import List
from gpiozero import OutputDevice, Button
from time import sleep

'''
--- LOGICA DI IMPLEMENTAZIONE ---
Hardware: Il sensore è una barriera a infrarossi (IR), composta da un emettitore (un LED IR) e un ricevitore (un fotodiodo o fototransistor).
L'emettitore è sempre acceso, creando un fascio di luce invisibile. Il ricevitore rileva questo fascio.

Rilevamento: Quando un oggetto (come una zanzara) attraversa il fascio, interrompe la luce.
Il ricevitore rileva questa interruzione e cambia lo stato del suo pin GPIO.

Observer Pattern: La classe SensoreMovimento agisce come Subject (Soggetto). Non conosce i dettagli di chi deve avvisare.
Si limita a mantenere una lista di iscritti (gli Observer) e, quando il fascio viene interrotto, invoca il metodo notifica().

notifica(): Questo metodo scorre la lista degli iscritti e chiama il loro metodo aggiorna(), informandoli che è successo qualcosa.

gpiozero: Usiamo la libreria gpiozero per un controllo pulito ed efficiente. L'emettitore sarà un OutputDevice (o LED) sempre acceso.
Il ricevitore sarà un InputDevice (o Button), che può lanciare eventi automaticamente quando il suo stato cambia, senza bisogno di un ciclo di controllo continuo.
'''

class SensoreMovimento(ISubject):
    """
    Il Soggetto (Subject) del pattern Observer.
    Rappresenta un sensore a barriera infrarossi (emettitore/ricevitore).
    Il suo compito è rilevare l'interruzione del fascio IR e notificare 
    gli observer iscritti.
    """
    def __init__(self, emitter_pin: int, receiver_pin: int):
        """
        Costruttore della classe SensoreMovimento.

        Args:
            emitter_pin (int): Il pin GPIO (BCM) per il LED emettitore IR.
            receiver_pin (int): Il pin GPIO (BCM) per il fotodiodo/fototransistor ricevitore.
        """
        self.stato: str = "Inizializzazione"
        self._observers: List[IObserver] = [] # Lista privata degli observer

        try:
            # --- Setup Hardware ---
            # L'emettitore è un semplice output che accendiamo e lasciamo così.
            self._emitter = OutputDevice(emitter_pin, initial_value=True)
            
            # Il ricevitore è un input. Usiamo la classe Button che è event-driven.
            # pull_up=True significa che il pin è HIGH (1) quando riceve luce
            # e va LOW (0) quando il fascio è interrotto.
            self._receiver = Button(receiver_pin, pull_up=True)
            
            # --- Collegamento Evento-Azione ---
            # Colleghiamo l'evento "pin va a LOW" (fascio interrotto) direttamente
            # al nostro metodo di notifica. Questo è il cuore dell'automatismo.
            self._receiver.when_pressed = self.notifica
            
            self.stato = "Operativo"
            print(f"Sensore IR inizializzato (Emettitore: {emitter_pin}, Ricevitore: {receiver_pin}). Stato: {self.stato}")

        except Exception as e:
            self.stato = "Errore"
            print(f"ATTENZIONE: Impossibile inizializzare GPIO. Il codice funzionerà in modalità 'virtuale'.")
            print(f"Dettagli: {e}")

    def iscrivi(self, observer: IObserver):
        """Aggiunge un observer alla lista degli iscritti."""
        if observer not in self._observers:
            self._observers.append(observer)
            print(f"Observer '{type(observer).__name__}' iscritto al sensore.")
        else:
            print(f"Observer '{type(observer).__name__}' è già iscritto.")

    def disiscrivi(self, observer: IObserver):
        """Rimuove un observer dalla lista."""
        try:
            self._observers.remove(observer)
            print(f"Observer '{type(observer).__name__}' disiscritto dal sensore.")
        except ValueError:
            print("Tentativo di disiscrivere un observer non iscritto.")

    def notifica(self):
        """
        Notifica a tutti gli observer che il fascio è stato interrotto.
        Questo metodo viene chiamato AUTOMATICAMENTE da gpiozero.
        """
        print(f"\n🔔 RILEVAMENTO! Fascio interrotto. Notifica in corso a {len(self._observers)} observer...")
        self.stato = "Rilevamento"
        
        for observer in self._observers:
            observer.aggiorna()
            
        self.stato = "Operativo" # Torna operativo dopo la notifica

    def _controllaMovimento(self):
        """
        Metodo concettuale. Nell'implementazione con gpiozero, non è necessario
        chiamarlo in un ciclo, poiché gli eventi sono gestiti automaticamente.
        """
        print("Il controllo del movimento è gestito da eventi hardware, non da un ciclo attivo.")
        pass



'''
# --- TEST ---
if __name__ == '__main__':
    # Questo blocco mostra come SensoreMovimento e un finto Controllore interagiscono.
    
    # 1. Definiamo un finto Controllore che agisce da Observer per il test
    class FintoControllore(IObserver):
        def aggiorna(self):
            print(">>> Controllore: Notifica ricevuta! Avvio la gestione del rilevamento...")
            # Qui il vero controllore inizierebbe a scattare foto, ecc.
            
    # 2. Creazione degli oggetti
    print("--- Inizio test del Sensore di Movimento (Observer Pattern) ---")
    sensore_ir = SensoreMovimento(emitter_pin=17, receiver_pin=18)
    controllore_trappola = FintoControllore()
    
    # 3. Il Controllore si iscrive al Sensore
    sensore_ir.iscrivi(controllore_trappola)
    
    # 4. Simulazione
    print("\nIl sistema è ora in attesa. Simuleremo l'interruzione del fascio tra 3 secondi...")
    print("(Per fermare, premi Ctrl+C)")
    
    try:
        # In un programma reale, il codice semplicemente attenderebbe qui.
        # L'evento verrebbe scatenato da un oggetto fisico.
        # Per questo test, simuliamo l'evento manualmente.
        sleep(3)
        if hasattr(sensore_ir, '_receiver') and sensore_ir._receiver:
            print("\n>>> SIMULAZIONE MANUALE: Interruzione del fascio <<<")
            # Simuliamo la pressione del "bottone" (il pin che va a LOW)
            sensore_ir._receiver.pin.drive_low() 
            sleep(0.1) # L'interruzione è breve
            sensore_ir._receiver.pin.drive_high()
        else:
            print("\nEsecuzione in modalità virtuale. Chiamo notifica() direttamente.")
            sensore_ir.notifica()
            
        sleep(2)
        print("\nSimulazione terminata. Il sistema continua ad essere in ascolto.")
        # Lasciamo lo script in esecuzione per un po' per mostrare che è in attesa
        sleep(10)
        
    except KeyboardInterrupt:
        print("\nProgramma terminato dall'utente.")
    finally:
        print("--- Test terminato ---")

'''
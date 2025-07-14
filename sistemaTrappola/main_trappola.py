# Data: 14 luglio 2025
# Autore: Mattia Muraro

from __future__ import annotations
from sensoreMovimento import SensoreMovimento
from controlloreTrappola import ControlloreTrappola
from moduloCAM import ModuloCAM
from modelloML import ModelloML
from led import LED
from gate import GATE
from ventola import VENTOLA

# --- 3. Esempio di utilizzo della struttura ---
if __name__ == '__main__':
    # Questo blocco mostra come le classi verrebbero collegate insieme
    
    # 1. Creazione dei componenti hardware e software
    cam1 = ModuloCAM()
    modello_principale = ModelloML()
    led_array = [LED()]
    gate_principale = [GATE()]
    ventola_aspirazione = [VENTOLA()]

    # 2. Creazione del SOGGETTO (il sensore) e dell'OSSERVATORE (il controllore)
    sensore = SensoreMovimento()
    controllore = ControlloreTrappola(
        lista_cam=[cam1],
        lista_modelli_ml=[modello_principale],
        lista_led=led_array,
        lista_gate=gate_principale,
        lista_ventole=ventola_aspirazione
    )
    
    # 3. ISCRIZIONE: Il Controllore si registra per ricevere notifiche dal Sensore
    print("Il Controllore si iscrive al Sensore...")
    sensore.iscrivi(controllore)
    
    # 4. SIMULAZIONE: Il Sensore rileva un movimento e notifica i suoi iscritti
    print("Il Sensore rileva un movimento e invia la notifica...")
    sensore.notifica() # Questo, nell'implementazione reale, chiamerebbe controllore.aggiorna()
    
    print("\nStruttura creata e connessione tramite Observer Pattern dimostrata.")
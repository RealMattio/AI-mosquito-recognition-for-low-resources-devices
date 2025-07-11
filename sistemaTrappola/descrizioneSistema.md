### **Report di Progettazione Software: Sistema Trappola**

Questo documento rappresenta la progettazione finale del "Sistema Trappola", consolidando le iterazioni di design in un blueprint architetturale completo. Vengono analizzate le tre prospettive UML complementari — strutturale (Classi), architetturale (Componenti) e comportamentale (Stati) — per fornire una guida esaustiva all'implementazione.

---
### 1. Diagramma delle Classi - La Struttura Statica e Disaccoppiata 🏗️

Il Diagramma delle Classi definisce il vocabolario statico del sistema, modellando i tipi di entità, le loro proprietà, i loro comportamenti e le relazioni strutturali che li legano.

![Diagramma delle Classi](./UML_graphs/diagramma%20delle%20classi%20aggiornato.svg)

#### **Analisi della Struttura**
L'architettura si basa su un design **disaccoppiato** e guidato da contratti formali. Al centro rimane la classe `ControlloreTrappola`, ma la sua interazione con i moduli di servizio avviene tramite dipendenze dirette da classi concrete. Sono state utilizzate delle **interfacce** (`ICamera`, `IPredizione`, `ISensoreMovimento`) che definiscono i contratti di servizio.

Le classi concrete (`ModuloCAM`, `ModelloML`, `SensoreMovimento`) **realizzano** (implementano) queste interfacce. Di conseguenza, il `ControlloreTrappola` dipende unicamente da queste astrazioni, aderendo al **Principio di Inversione delle Dipendenze**. Le relazioni con gli attuatori (`LED`, `GATE`, `VENTOLA`) sono di **Composizione**, indicando un forte legame di possesso.

#### **Punti Chiave del Design**
* **Architettura Disaccoppiata:** La dipendenza da interfacce è il maggior punto di forza del design, poiché rende il sistema flessibile, sostituibile e facilmente testabile.
* **Contratti Formali:** Le interfacce stabiliscono API chiare tra le diverse parti del software.
* **Incapsulamento:** La logica interna del controllore rimane nascosta, esponendo solo i metodi pubblici necessari.

---
### 2. Diagramma delle Componenti - L'Architettura Fisica e Modulare 📦

Il Diagramma dei Componenti illustra la partizione del sistema in moduli software coesivi e sostituibili, definendo l'architettura fisica e le dipendenze di alto livello.

![Diagramma delle Componenti](./UML_graphs/diagramma%20delle%20componenti.svg)

#### **Analisi della Struttura**
Il sistema è modularizzato in componenti come `SistemaDiControllo`, `ServizioFotocamera` e `DriverAttuatori`. Questo diagramma ha guidato il miglioramento del Diagramma delle Classi, introducendo fin da subito il concetto di comunicazione basata su interfacce.

#### **Punti Chiave del Design**
* **Design Modulare:** La suddivisione in componenti permette lo sviluppo parallelo e la manutenzione isolata di ogni parte del sistema.
* **Comunicazione tramite Interfacce:** Questo principio architetturale è pienamente riflesso nel Diagramma delle Classi. Il `SistemaDiControllo` richiede l'interfaccia `ICamera`, che viene fornita dal componente `ServizioFotocamera`. Questo allineamento tra i due diagrammi è fondamentale per un design robusto.

---
### 3. Diagramma a Stati - Il Comportamento Dinamico e Intelligente ⚡

Il Diagramma a Stati descrive il ciclo di vita del sistema, specificando come esso reagisce agli eventi e transita tra diverse condizioni operative.

![Diagramma a Stati](./UML_graphs/diagramma%20a%20stati.svg)

#### **Analisi della Struttura**
Il flusso comportamentale finale è stato raffinato per includere una logica di verifica a due passaggi, che ne aumenta l'intelligenza e l'efficienza:
1.  **Stato `InAttesa`:** Una prima valutazione a basso consumo viene eseguita dalla `camera1`.
2.  **Stato `ConfermaProcedura`:** Solo se un potenziale target viene rilevato, il sistema entra in questo stato, attivando la `camera2` per un'analisi più accurata e una decisione finale.
3.  **Procedure Finali:** A seconda della conferma, il sistema attiva la `ProceduraCatturaFinale` o la `ProceduraRepulsioneFinale`, oppure ritorna in attesa in caso di fuga dell'insetto.
4.  **Ciclo Chiuso:** Ogni procedura, una volta completata, riporta il sistema allo stato `InAttesa`, garantendo che sia sempre pronto per un nuovo ciclo.

#### **Punti Chiave del Design**
* **Logica di Verifica a Due Passaggi:** L'introduzione dello stato `ConfermaProcedura` rende il sistema più efficiente e meno prono a errori, attivando le risorse più dispendiose solo quando necessario.
* **Flusso di Controllo Robusto:** Il comportamento del sistema è deterministico e chiaramente definito per ogni evento possibile in ogni stato.
* **Azioni di Ingresso (`entry`):** Le azioni sono legate agli stati, semplificando la logica delle transizioni e garantendo che le operazioni di inizializzazione vengano eseguite in modo consistente.

---
### 4. Visione d'Insieme e Integrazione

I tre diagrammi formano un modello coeso e completo del "Sistema Trappola":
* Le **classi** e le **interfacce** del Diagramma delle Classi sono gli elementi costitutivi che vengono assemblati nei **componenti** del Diagramma dei Componenti.
* I **metodi** definiti nelle classi (`ControlloreTrappola.apriGate()`) sono le **azioni** eseguite durante le transizioni o all'ingresso degli **stati** nel Diagramma a Stati.
* Gli **eventi** che guidano il Diagramma a Stati (`movimentoRilevato`) sono generati dall'esecuzione di **metodi** all'interno delle classi (`SensoreMovimento.informaHead()`).

Questo approccio integrato garantisce che l'architettura, la struttura del codice e il comportamento dinamico siano perfettamente allineati, fornendo un blueprint solido e affidabile per l'implementazione.
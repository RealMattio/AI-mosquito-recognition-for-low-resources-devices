### **Report di Progettazione Software: Sistema Trappola**

Questo documento rappresenta la progettazione finale del "Sistema Trappola", consolidando le iterazioni di design in un blueprint architetturale completo. Vengono analizzate le tre prospettive UML complementari — strutturale (Classi), architetturale (Componenti) e comportamentale (Stati) — per fornire una guida esaustiva all'implementazione.

-----

### 1\. Diagramma delle Classi - La Struttura Statica e Disaccoppiata 🏗️

Il Diagramma delle Classi definisce il vocabolario statico del sistema, modellando i tipi di entità, le loro proprietà, i loro comportamenti e le relazioni strutturali che li legano.

#### **Analisi della Struttura**

L'architettura si basa su un design **disaccoppiato** e guidato da contratti formali. Al centro rimane la classe `ControlloreTrappola`, la cui interazione con i moduli di servizio avviene tramite le **interfacce** (`ICamera`, `IPredizione`) che definiscono i contratti.

Le classi concrete (`ModuloCAM`, `ModelloML`, `SensoreMovimento`) **realizzano** (implementano) queste interfacce e i pattern specifici. Di conseguenza, il `ControlloreTrappola` dipende unicamente da astrazioni, aderendo al **Principio di Inversione delle Dipendenze**. Le relazioni con gli attuatori (`LED`, `GATE`, `VENTOLA`) sono di **Composizione**, indicando un forte legame di possesso.

#### **Punti Chiave del Design**

  * **Architettura Disaccoppiata:** La dipendenza da interfacce è il maggior punto di forza del design, poiché rende il sistema flessibile, sostituibile e facilmente testabile.
  * **Contratti Formali:** Le interfacce stabiliscono API chiare tra le diverse parti del software.
  * **Incapsulamento:** La logica interna del controllore rimane nascosta, esponendo solo i metodi pubblici necessari.

#### **Pattern Architetturale Adottato: Observer Pattern**

Per gestire la comunicazione tra il `SensoreMovimento` (la fonte dell'evento) e il `ControlloreTrappola` (l'ascoltatore), è stato implementato l'**Observer Pattern**.

Invece di avere un sensore che conosce e chiama direttamente il controllore, il pattern inverte questa logica:

  * Il **Sensore** (`Subject`) mantiene una lista di oggetti interessati (`Observers`) e li notifica in modo generico quando rileva un movimento.
  * Il **Controllore** (`Observer`) si iscrive al sensore per ricevere queste notifiche e reagire di conseguenza.

**Vantaggi ottenuti:**

  * **Disaccoppiamento:** Il sensore non sa nulla del controllore, promuovendo l'indipendenza dei moduli.
  * **Estensibilità:** È possibile aggiungere nuovi "ascoltatori" (come un logger o un sistema di allarme) senza modificare il codice del sensore.
  * **Flessibilità:** Le relazioni tra chi notifica e chi viene notificato possono essere stabilite e modificate dinamicamente.

-----

### 2\. Diagramma delle Componenti - L'Architettura Fisica e Modulare 📦

Il Diagramma dei Componenti illustra la partizione del sistema in moduli software coesivi e sostituibili, definendo l'architettura fisica e le dipendenze di alto livello.

#### **Analisi della Struttura**

Il sistema è modularizzato in componenti come `SistemaDiControllo`, `ServizioFotocamera` e `DriverAttuatori`. Questo diagramma ha guidato il miglioramento del Diagramma delle Classi, introducendo fin da subito il concetto di comunicazione basata su interfacce.

#### **Punti Chiave del Design**

  * **Design Modulare:** La suddivisione in componenti permette lo sviluppo parallelo e la manutenzione isolata di ogni parte del sistema.
  * **Comunicazione tramite Interfacce:** Questo principio architetturale è pienamente riflesso nel Diagramma delle Classi. Il `SistemaDiControllo` richiede l'interfaccia `ICamera`, che viene fornita dal componente `ServizioFotocamera`. Questo allineamento tra i due diagrammi è fondamentale per un design robusto.

-----

### 3\. Diagramma a Stati - Il Comportamento Dinamico e Intelligente ⚡

Il Diagramma a Stati descrive il ciclo di vita del sistema, specificando come esso reagisce agli eventi e transita tra diverse condizioni operative.

#### **Analisi della Struttura**

Il flusso comportamentale finale è stato raffinato per includere una logica di verifica a due passaggi, che ne aumenta l'intelligenza e l'efficienza:

1.   **Stato `InAttesa`:** Una prima valutazione a basso consumo viene eseguita dalla `camera1`.
2.   **Stato `ConfermaProcedura`:** Solo se un potenziale target viene rilevato, il sistema entra in questo stato, attivando la `camera2` per un'analisi più accurata e una decisione finale.
3.   **Procedure Finali:** A seconda della conferma, il sistema attiva la `ProceduraCatturaFinale` o la `ProceduraRepulsioneFinale`, oppure ritorna in attesa in caso di fuga dell'insetto.
4.   **Ciclo Chiuso:** Ogni procedura, una volta completata, riporta il sistema allo stato `InAttesa`, garantendo che sia sempre pronto per un nuovo ciclo.

#### **Punti Chiave del Design**

  * **Logica di Verifica a Due Passaggi:** L'introduzione dello stato `ConfermaProcedura` rende il sistema più efficiente e meno prono a errori, attivando le risorse più dispendiose solo quando necessario.
  * **Flusso di Controllo Robusto:** Il comportamento del sistema è deterministico e chiaramente definito per ogni evento possibile in ogni stato.
  * **Azioni di Ingresso (`entry`):** Le azioni sono legate agli stati, semplificando la logica delle transizioni e garantendo che le operazioni di inizializzazione vengano eseguite in modo consistente.

-----

### 4\. Visione d'Insieme e Integrazione

I tre diagrammi formano un modello coeso e completo del "Sistema Trappola":

  * Le **classi** e le **interfacce** del Diagramma delle Classi sono gli elementi costitutivi che vengono assemblati nei **componenti** del Diagramma dei Componenti.
  * I **metodi** definiti nelle classi (`ControlloreTrappola.apriGate()`) sono le **azioni** eseguite durante le transizioni o all'ingresso degli **stati** nel Diagramma a Stati.
  * Gli **eventi** che guidano il Diagramma a Stati (`movimentoRilevato`) sono generati dall'esecuzione di **metodi** all'interno delle classi (`SensoreMovimento.notifica()`).

Questo approccio integrato garantisce che l'architettura, la struttura del codice e il comportamento dinamico siano perfettamente allineati, fornendo un blueprint solido e affidabile per l'implementazione.

***

### 5\. Implementazioni Future e Pattern Evolutivi 🚀

Per tradurre il design attuale in codice ancora più robusto, manutenibile e flessibile, si consiglia l'adozione dei seguenti design pattern durante la fase di implementazione.

#### **State Pattern**
* **Perché usarlo:** Per tradurre il Diagramma a Stati in codice object-oriented. Invece di usare un grande costrutto `if/else` o `switch` all'interno del `ControlloreTrappola` per gestire lo stato corrente, questo pattern incapsula il comportamento di ogni stato (`InAttesa`, `ConfermaProcedura`, etc.) in una classe separata.
* **Vantaggi:**
    * **Codice Pulito:** Elimina la logica condizionale complessa dal controllore.
    * **Estensibilità:** Aggiungere un nuovo stato significa semplicemente creare una nuova classe, senza modificare quelle esistenti (aderendo al Principio Open/Closed).

#### **Strategy Pattern**
* **Perché usarlo:** Per gestire gli algoritmi delle procedure finali (`ProceduraCatturaFinale` e `ProceduraRepulsioneFinale`). Invece di codificare la logica di cattura e repulsione direttamente nel controllore o in uno stato, questo pattern le incapsula in "strategie" separate e intercambiabili.
* **Vantaggi:**
    * **Flessibilità:** Permette di aggiungere nuove procedure (es. una "strategia di rilascio") o modificare quelle esistenti senza toccare la logica del controllore.
    * **Separazione delle Responsabilità:** Gli algoritmi sono separati dal contesto che li utilizza.

#### **Factory Method Pattern**
* **Perché usarlo:** Per gestire la creazione degli oggetti concreti (come `ModuloCAM` o `ModelloML`) di cui il `ControlloreTrappola` ha bisogno. Invece di istanziare direttamente una `new ModuloCAM()`, il controllore si affiderebbe a una "fabbrica" per ottenere l'oggetto corretto che implementa l'interfaccia `ICamera`.
* **Vantaggi:**
    * **Disaccoppiamento:** Il codice principale non è più accoppiato alle classi concrete, ma solo alle loro interfacce e alla fabbrica.
    * **Configurabilità:** Semplifica la creazione di diverse configurazioni della trappola (es. una versione "pro" con sensori avanzati e una "base") semplicemente utilizzando una fabbrica diversa.
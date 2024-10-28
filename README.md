Cartella tutorial: 
  - client.py :  codice ralativo ai client, viene definita la classe client, la funzione di fit ed evaluation
  - server.py: qui vengono definite la strategia, server address, eventuali metodi per la valutazione del modello server side
  - fedxgb_bagging.py : strategia usata per fare addestramento federato
  - datset.py : analisi dei dataset utilizzati
    
file main.py:
  - in questo file è presente il codice per creare le partizioni del dataset , creare le dmatrix e per creare i client e valutare modello federato (base model),          centralized model e local calibration. (utilizzo dataset Nf-Ton_Iot-V2)
    
file main copy.py:
  - come file main.py ma relativo al dataset CICIDS2017.

file MyFedAvg.py:
  - file in cui è implementata la federazione del calibratore, relativo a Nf-Ton_Iot-V2

file MyFedAvg copy.py:
  - file in cui è implementata la federazione del calibratore, relativo a CICIDS2017

Cartella flwr/server:
  - codice relativo al server già definito da flower, presenta alcune modifiche tutte commentate non influenzano il funzionamento. unica cosa da sottolineare riga         173 fino a riga 177, codice inserito per poter salvare il modello globale una volta finito l'addestramento federato. questa parte va commentata quando si vuole       addestrare il calibratore in maniera federata (quindi quando si usa il file MyFedAvg.py)
nota:
  in client.py e server.py sono implementati anche la logica e alcune funzioni per poter fare one shot federated learning. (risultati non buoni)



FUNZIONAMENTO:
l'idea è creare prima le partizioni dei dati, poi addestrare il modello federato, quindi valutare le prestazioni del modello federato, del calibratore centralizzato e le prestazioni del calibratore custom sui client isolati. 

Bisogna usare 2 terminal: 
  -  nel primo accedere alla cartella tutorial e chiamre il file server.py.
  -  nel secondo terminal, chiamare il file main.py oppure main copy.py in base al dataset che si vuole utilizzare.
      - ricordarsi di aggiornare il valore di alpha nell'algoritmo per le partizioni e impostare correttamente il percorso file per salvare le Dmatrix.

completate queste operazioni si passa alla valutazione del calibratore federato:

ricordasi di commentare le righe 173 - 177 del file server.py nella cartella flwr/server. poi da terminal chiamare MyFedAvg.py. 

(qui ricordarsi il valore di alpha da settare per leggere le partizioni corrette, impostare learning rate, numero di epoche globali e locali per addestramento)

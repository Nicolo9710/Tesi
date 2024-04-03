Cartella tutorial: 
  - client.py :  codice ralativo ai client, viene definita la classe client, la funzione di fit ed evaluation
  - server.py: codice relativo al server (è presente un messaggio di warning quando viene eseguita la valutazione centralizzata)
  - fedxgb_bagging.py : strategia usata per fare addestramento federato
  - datset.py : ancora da finire, dove viene caricato il dataset e vengono fatte considerazioni rispetto alla distribuzione delle classi delle partizioni create

file main.py:
  - in questo file è presente il codice per creare le partizioni (bilanciate) del dataset , creare le dmatrix e per creare i client

file main_sbilanciato.py:
  - in questo file è presente il codice per creare le partizioni (sbilanciate) del dataset , creare le dmatrix e per creare i client

file prova_random_forest.ipynb: (ancora da sistemare):
  - scorrere fino in fondo il file, e considerare solo la sezione prova con dask, qui viene creata una random_forest "centralizzata", sono poi eseguiti alcuni test per cercare di trovare i parametri migliori,
    vengono poi caricati e valutati 2 modelli addestrati utilizzando tecnica federata con 10 client, uno con distribuzione del dataset tra i client bilanciata, e uno con distribuzione dataset tra i client sbilanciata.

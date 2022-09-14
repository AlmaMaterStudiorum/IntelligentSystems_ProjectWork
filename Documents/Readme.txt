
Lo scopo di questa attività di progetto è ricercare delle tecniche di compressioni di reti neurali per ridurre la dimensione 
e la velocità di esecuzione delle reti in modalità inferenza e contemporaneamente valutare l'impatto della differenza tra la versione di base nelle reti con la versione compressa 
quando è inserita in un problema di ottimizzazione combinatoria utilizzando la tecnica EML.
La ricerca si è concentrata sulle Tensorflow Model Optimization Toolkit , in particolar sulle api di compressione (tflite) e su quelle di pruning.
Il fattore di compressione in tflite è buono (???) ma non permette la sua codifica in un problema di ottimizzazione combinatoria attraverso le funzioni di libreria di EML, 
nello specifico la libreria util accetta modelli keras\h5 ma non modelli tflite in quanto mancano dei metodi di accesso ai layers e neurons.
Al contrario il pruning del modello di baseline è codificabile come modello di OC.
- Presentazione delle API di TMOT (TBD)
- Valori di configurazione (TBD)
- Architettura del progetto ,pipeline (TBD)
- Preparazione al progetto (SIR) (TBD)
- Summary delle metriche
- Risultati


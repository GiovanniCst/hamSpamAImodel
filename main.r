# Giovanni Costantini - auto formazione Machine Learning

# Installa le librerie che ci verranno utili
install.packages(c("ggplot2", "e1071", "caret", "quanteda", "irlba", "randomForest", "doSNOW"))

# Carica il dataset CSV e tienilo in memoria nella variabile spam.raw
spam.raw <- read.csv("spam.csv", stringsAsFactors = FALSE, fileEncoding = "UTF-16")

# Mostra il file utilizzando la funzionalità specifica di R Studio
View(spam.raw)

# Elimina le colonne in più che emergono dall'importazione tenendo solamente le prime due colonne
spam.raw <- spam.raw[, 1:2]

# Modifica i titoli di colonne in qualcosa di maggiormente significativo
names(spam.raw) <- c("Label", "Text")
View(spam.raw)

# Verica che il dataset sia completo
# ovvero: dammi la lunghezza (length) del vettore generato dalla funzione negata (!) complete.cases
length(which(!complete.cases(spam.raw)))

# Converti l'etichetta della classe in un fattore
spam.raw$Label <- as.factor(spam.raw$Label)

# Diamo un'occhiata alla distribuzione delle etichette sui dati table(spam.raw$Label) e, volendo vederlo in percentuale
# passiamo il risultato a prop.table()
prop.table(table(spam.raw$Label))

# Ritenendo che la lunghezza del massaggio nel dataset possa essere un fattore rilevante
# Estraggo questo valore con nchar e lo aggiungo al dataset come feature aggiuntiva
spam.raw$TextLength <- nchar(spam.raw$Text)

# Ora fammi vedere delle statistiche relative a questa nuova colonna (così mi rendo conto della distribuzione dei valori.)
# Fai caso alla differenza tra Min e Max, al valore Median (ovvero l'entry che separa il 50% dei numeri più bassi dal 50%
# dei numeri più alti e la differenza di questo con il valore di media (Mean))
summary(spam.raw$TextLength)

# Volendo visualizzare i dati, utilizzo ggplot2. Con library() carico e collego il pacchetto
library(ggplot2)

# Preparo la visualizzazione differenziando tra le due etichette ham/spam. fill = Label colora le barre con le etichette
# binwidth è la larghezza delle barre
ggplot(spam.raw, aes(x=TextLength, fill = Label)) +
  theme_bw() +
  geom_histogram(binwidth = 5) +
  labs(y="Numero dei messaggi", x="Numero di caratteri per messaggio", 
       title = "Distribuzione della lunghezza dei Messaggi con suddivisione per Etichette delle classi")

# Ora si procede allo split dei dati per riservarne una parte per il training ed un'altra parte per il testing del modello. Il pacchetto
# caret serve a questo. La suddivisione tipica è 70/30%
library(caret)
help(package = "caret")

# Utilizziamo caret per procedere allo splitting 70/30%. In considerazione del fatto che la proporzione delle etichette è 
# sbilanciata (86.593% con etichetta HAM e solo 13.4063 con etichetta SPAM) prenderemo dei provvedimenti 
# [passando alla funzione createDataPartition di caret le etichette] affinchè negli split si ripetano le 
# medesime proporizioni (tecnicamente definito "stratified split").
# In primis settiamo il seed del random number generator per ottenere i medesimi risultati pseudo random del tutorial
set.seed(32984)

# createDataPartition:
# spam.row&Label gli consente di creare uno split che presenti le medesime proporzioni dei dati originali
# con times = 1 gli indico che voglio una sola divisione
# con p = 0.7 gli dico che voglio un campione del 70% dei dati
# con list = FALSE gli dico che non voglio ottenere la lista che genera di default, ma solamente gli indici che siano rappresentatvi
# di quanto chiedo
indexes <- createDataPartition(spam.raw$Label, times = 1, p = 0.7, list = FALSE)

# ora la variabile indexes contiene il riferimento (gli indici) ad un subset di dati corrispondenti al 70% dei dati originari,
# nelle medesime proporzioni tra HAM e SPAM come come nel dataset completo

# ora fitro le righe del dataset originale (spam.raw) prendendo quelle indicate in indexes cosìcchè nella variabile train
# vada il 70% proporzionato di cui sopra
train <- spam.raw[indexes,]

# ora fitro le righe del dataset originale (spam.raw) prendendo quelle indicate nell'opposto di indexes (ovvero quelle che NON sono
# in indexes). Ottengo l'opposto aggiungendo il segno - ad indexes
test <- spam.raw[-indexes,]

# Ricorro a prop.table() nuovamente per assicurarmi che le proporzioni siano mantenute
prop.table(table(spam.raw$Label))
prop.table(table(test$Label))
prop.table(table(train$Label))

# La libreria quanteda contiene una serie di metodi utili per l'analisi del testo. Procediamo a caricarla e collegarla

library(quanteda)
help("quanteda")

# I PASSAGGIO; Il primo passaggio prevede la tokenizzazione delle parole contenute nella colonna Text del nostro dataset 
# e la contestuale rimozione dei numeri, della punteggiatura, dei simboli e dei trattini. Il risultato dell'operazione
# viene memorizzato nella variabile train.tokens

train.tokens <- tokens(train$Text, what = "word",
                       remove_numbers = TRUE, remove_punct = TRUE,
                       remove_symbols = TRUE, remove_hyphens = TRUE)

# vediamo il risultato dell'operazione
train.tokens[357]

# II PASSAGGIO: Procediamo a trasformare le lettere maiuscole in minuscole
train.tokens <- tokens_tolower(train.tokens)

# vediamo il risultato dell'operazione
train.tokens[357]

# III PASSAGGIO: Rimuoviamo le stopwords. Per ottenere un elenco delle stopwords digitare stopwords() in console
# per vedere le stopwords di un linguaggio diverso dall'inglese, digitare es: stopwords(kind="italian")
train.tokens <- tokens_select(train.tokens, stopwords(), selection = "remove")

# vediamo il risultato dell'operazione
train.tokens[357]

# IV PASSAGGIO: Stemming - Riduzione delle parole alle relative radici
train.tokens <- tokens_wordstem(train.tokens, language = "english")

# vediamo il risultato dell'operazione
train.tokens[357]

# Il preprocessing in 4 passaggi è ora terminato.

# Creazione del primo modello "Cesto delle parole" "Bag of words" [è un dfm, Data Frequency Matrix]. Il Cesto delle Parole
# crea una tabella in cui nelle colonne trovo il conteggio del numero delle ricorrenze di ciascun token (parola nel ns caso)
# all'interno di ciascun documento
train.tokens.dfm <- dfm(train.tokens, tolower = FALSE)

# Converto la dfm di cui sopra in un Matrice che è più semplice da gestire con la funzione as.matrix(). Trasformandola in una
# Matrice standard R, la posso anche vedere. 
train.tokens.matrix <- as.matrix(train.tokens.dfm)

# Mostrami i primi 20 documenti e le prime 100 colonne
View(train.tokens.matrix[1:20, 1:100])

# Dammi la dimensione della matrice
dim(train.tokens.matrix)

# Diamo un'occhiata ai nomi delle colonne (che sono i token, ovvero i termini contenuti nel dataset) che ci mostra anche il
# risultato dello stemming
colnames(train.tokens.matrix)[1:50]

# Procediamo a costruire il nostro modelli, utilizzando le metodologia della cross validation, che ci consente di massimizzare l'uso
# dei nostri dati per il training ed avere una stima di come il modello si potrebbe comportare in produzione.
# Utilizzare la cross validation è critico per ottenere una stima sulle perfomance in real life del nostro modello.

# Per procedere alla cross validation occorrono due cose:
# 1. Le features che abbiamo creato - ovvero la Document Frequency Matrix creata poco sopra. Come già detto la DFM è una rappresentazione
# in cui ciasuna riga è un documento ed ogni colonna è un token, ovvero una parola. L'incrocio tra token è ducumento contiene
# il conteggio di quante volte quella parola compare in quel documento del corpus

# as.data.frame trasforma il nostro train.token.dfm in un Data Frame comprensibile al linguaggio R (primo ingrediente)

# 2. Volendo creare un modello che riconosce email di spam da email legittime, il sencondo ingrediente che ci serve sono le 
# etichette ham e spam che prendo dal dataset train. Le etichette sono l'elemento di feedback

# La funzione di cbind accoppia i due ingredienti assieme ed salva il risultato nella variabile train.tokens.df

train.tokens.df <- cbind(Label = train$Label, as.data.frame(train.tokens.dfm))

# Possono ancora esistere impurità nei dati:
names(train.tokens.df)[c(145, 146, 234, 237)]

# I nomi colonne non possono contenere punteggiatura o numeri, e per evitare errori dopo, procediamo a pulirli utilizzando la
# funzione make.names
names(train.tokens.df) <- make.names(names(train.tokens.df))

# Il nostro modello sarà costruito tramite statified cross validation. Dobbiamo anche tenere in considerazione che classi dei nostri dati 
# ham e spam sono sbilanciate (molto ham e poco spam). Quindi i campioni random esaminati dall'algoritmo dovranno presentare
# proporzioni di classi coerenti.
# Utilizziamo funzionalità della libreria caret per creare stratified folds for 10-fold cross validation repeated 3 times
# (ovvero crea 30 campioni stratificati). Ripetere l'operazione tre volte aumenta la posibilità di avere stime più valide/accurate,
# Tanto più quando siamo di fronte a dati con classi sbilanciate, come nel nostro caso.

# Come sopra, settiamo il seed del random generator per ottenere i medesimi risultati del tutorial
set.seed(48743)

# La funzione createMultiFolds() crea campioni di dati stratificati multipli. createMultiFolds contiene funzionalità di
# createDataPartitions che abbiamo utilizzato all'inizio per splittare i nostri dati tra train e test.
# A createMultiFolds passiamo le classi (ovvero le etichette che prendiamo dai train data). Questo gli consente di creare 
# campioni di dati proporzionati in termini di distribuzione statistica delle classi (etichette)

cv.folds <- createMultiFolds(train$Label, k = 10, times = 3)

# Con l'istruzione che segue specifico come voglio che avvenga il training. Il metodo che richiediamo è Repeated Cross Validation,
# il numero di folds è 10, il numero di ripetizioni è 3. Avendo richiesto una Cross validation stratificata, dobbiamo indicare
# i folds tramite parametro index. Strafied significa "con le classi in proporzione"
cv.cntrl <- trainControl(method = "repeatedcv", number = 10,
                         repeats = 3, index = cv.folds)


# In considerazione che la nostra matrice è costituita da 22 milioni di elementi, c'e' la necessità di ottimizzare lanciando
# il training in parallelo. Questa funzionalità viene implementata tramite il pacchetto doSNOW. Procediamo a caricarla e collegarla
library(doSNOW)

# Controlla il tempo di esecuzione dell'operazione
start.time <- Sys.time()

# Crea un cluser che lavori su 7 core logici - ovvero lancia 7 istanze di R-Studio e consente a carot di utilizzarle contemporaneamente
# per l'elaborazione (è una semplificazione ma funziona). 
cl <- makeCluster(3, type = "SOCK")

# Questa funzione informa carot che le istanze di R-Studio sono ora disponibili per essere utilizzate
registerDoSNOW(cl)

# E' arrivato il momento di utilizzare la funzione di training dell' AI. Visto che attualmente i nostri dati hanno una dimensione
# importante, utilizziamo un algoritmo a single decision tree per il nostro training. In un secondo momento, quando applicheremo
# delle teniche di estrazione delle features che ridurranno la dimensione dei nostri dati, ci sposteremo su algoritmi più
# performanti

# Il paramentro method definsce quale modello di machine learning vogliamo creare - nel nostro caso, un rpart (che è un single decision tree)
# L'istruzione Label ~ . dice prevedi la Label utilizzando tutto il resto dei dati
# Il paramentro data = train.tokens.df indica dove prendere i dati, nel nostro caso nel dataframe train.tokens.df
# Il parametro trControl = cv.cntrl indica quale processo seguire - lo abbiamo definito poco sopra
# Il paramentro tuneLength = 7 indica a R di provare 7 differenti configurazioni dell' algoritmo rpart, verificare quale conf funzioni meglio ed utilizzare quella

# Si noti la separazione netta tra il processo trControl = cv.cntrl dal modello rpart

rpart.cv.1 <- train(Label ~ ., data = train.tokens.df, method = "rpart",
                    trControl = cv.cntrl, tuneLength = 7)

# Elaborazione completata, chiudi i cluster
stopCluster(cl)

# Calcola quanto tempo c'è voluto per il training
total.time <- Sys.time() - start.time
total.time

# Controlla i risultati
rpart.cv.1

#### Sopra: codice fino alla lezione 4 : https://youtu.be/IFhDlHKRHno


# L'utilizzo delle funzioni Term-Frequency (TF) ed Inverse Document Frequency (DF)
# è una tecnica potente per migliorare le informazioni/i segnali
# contenuti nella nostra matrice Document-Frequency. Specificatamente
# la matematica che sta dietro alle funzioni di TF-IDF raggiunge gli scopi che seguono:
#
#   1 - Il calcolo del Term Frequency (TF) tiene conto del fatto che documenti
#       più lunghi produrranno un conteggio di parole più alto. Applicando
#       la funzione di TF, questa normalizzerà tutti i documenti, passando
#       da un conteggio di tipo numerico ad una rappresentazione percentuale rendendo
#       quindi irrilevante la lunghezza del documento stesso.
#   2 - Il calcolo dell' Inverse Document Frequency tiene conto delle occorrenze
#       di ciascun termine in tutti i documenti. Più un termine appare nei diversi
#       documenti, meno questo termine ci sarà utile per distinguere un documento
#       dall' altro.
#   3 - Moltiplicare TF per IDF in ciascuna cella della matrice consente di soppesare e quindi
#       modificare i valori di ciascuna cella in funzione dei fattori 1 e 2 sopra.


# Definisco la funzione per il calcolo del Term Frequency TF:
term.frequency <- function(row) {
  row / sum(row)
}

# Definisco la funzione per il calcolo del Inverse Document Frequency
inverse.doc.freq <- function(col) {
  corpus.size <- length(col)
  doc.count <- length(which(col > 0))
  log10(corpus.size/doc.count)
}

# Definisco la funzione TF-IDF
tf.idf <- function(tf, idf) {
  tf * idf
}

# Ora applico la funzione di normalizzazione TF a tutti i documenti
# 1 sta ad indicare di applicare la funzione sulle righe, dopo l'uno compare la funzione
train.tokens.df <- apply(train.tokens.matrix, 1, term.frequency)

# Mostrami la dimensione della matrice
dim(train.tokens.df)

# Visualizza la matrice in R-Studio - nota che è ruotata!!! Documenti nelle colonne, termini sulle righe
View(train.tokens.df)[1:20, 1:100]


# Ora calcolo il vettore IDF che utilizzeremo sia per i dati di training che per i dati di test
train.tokens.idf <- apply(train.tokens.matrix, 2, inverse.doc.freq)

# Mostrami la struttura dell'oggetto
str(train.tokens.idf)

# In ultimo applica la funzione TF-IDF al nostro corpus
train.tokens.tfidf <- apply(train.tokens.df, 2, tf.idf, idf = train.tokens.idf)
dim(train.tokens.tfidf)
View(train.tokens.tfidf)[1:25, 1:25]

# Ora dobbiamo procedere a riportare la matrice al suo stato originario, ovvero con i documenti
# sulle righe ed i token sulle colonne, utilizzando la funzione di transpose t(). La matrice è stata
# ruotata quando abbiamo applicato la funzione di normalizzazione di Term Frequency poco sopra

# Al momento è così:
dim(train.tokens.tfidf)

# Ruoto
train.tokens.tfidf <- t(train.tokens.tfidf)

# Diventa così
dim(train.tokens.tfidf)

# Mostramela
View(train.tokens.tfidf)[1:25, 1:25]

# Controlla se ci sono dati incompleti
incomplete.cases <- which(!complete.cases(train.tokens.tfidf))
train$Text[incomplete.cases]

# Sistema gli incomplete cases sostituendoli con 0.0

train.tokens.tfidf[incomplete.cases,] <- rep(0.0, ncol(train.tokens.tfidf))
dim(train.tokens.tfidf)

# Verifica che non esistano più celle con dati invalidi
sum(which(!complete.cases(train.tokens.tfidf)))

# Costruiamo il data frame utilizzando il processo utilizzato in precedenza:
train.tokens.tfidf.df <- cbind(Label = train$Label, data.frame(train.tokens.tfidf))
names(train.tokens.tfidf.df) <- make.names(names(train.tokens.tfidf.df))

#### Sopra: codice fino alla lezione 5 : https://youtu.be/az7yf0IfWPM









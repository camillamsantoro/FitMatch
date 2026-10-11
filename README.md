# Recommender ibrido di programmi di allenamento

Progetto per il corso di **Sistemi Intelligenti per Internet** — Università degli Studi
Roma Tre. Autrice: Camilla Maria Santoro.

## In breve

Una persona dice cosa cerca (obiettivo, livello, attrezzatura, giorni e minuti a
disposizione) e il sistema le propone i tre programmi più adatti fra **2.671 programmi
reali** (2.699 nel dataset, 28 esclusi dalla pulizia).

Ogni programma riceve due punteggi:

- **rilevanza**: quanto corrisponde a ciò che la persona ha chiesto;
- **qualità**: quanto è costruito bene, misurato sul dettaglio degli esercizi (volume,
  frequenza, progressione, equilibrio muscolare, varietà, coerenza con l'obiettivo).

**Domanda di ricerca:** aggiungere la qualità migliora i consigli rispetto a un sistema
che usa solo i dati dichiarati e il testo?

Il dataset non contiene utenti né valutazioni, quindi il sistema è ibrido
(knowledge-based + content-based) e la valutazione usa **300 utenti simulati**, messi
alla prova con le 46 risposte di un questionario a persone vere.

## Risultati

NDCG@10 sui 300 utenti simulati (tabella completa in
`outputs/tables/valutazione_sistemi.csv`):

| Sistema | NDCG@10 |
|---|---|
| Casuale | 0,496 |
| Solo testo (il migliore dei tre: LSA) | 0,520 |
| Solo metadati | 0,706 |
| Rilevanza | 0,745 |
| **Rilevanza + qualità** | **0,783** |

- La qualità aggiunge **+0,038** sulla sola rilevanza, e il risultato regge in 8 scenari
  su 8 del simulatore e in 22 configurazioni di soglie su 22.
- Gran parte del guadagno si ottiene anche con un metadato dichiarato (il numero di
  esercizi): sopra di esso la qualità aggiunge +0,0115, sempre positivo ma significativo
  da 4 a 8 su 8 in 21 realizzazioni del rumore.
- Il guadagno viene quasi tutto dalla varietà, e si paga: i consigli oltre la capacità
  della persona salgono dall'11,3% al 18,1%.

## Dati

- **Dataset principale** — Kaggle, *600K+ Fitness Exercise & Workout Program Dataset*:
  <https://www.kaggle.com/datasets/adnanelouardi/600k-fitness-exercise-and-workout-program-dataset>.
  Scaricare `program_summary.csv` e `programs_detailed_boostcamp_kaggle.csv` e metterli
  in `archive/`. Al primo uso il codice controlla che siano i file originali.
- **free-exercise-db** — <https://github.com/yuhonas/free-exercise-db>, dominio pubblico:
  serve a sapere quale muscolo allena ogni esercizio. Si scarica da solo.
- **Questionario** — le 46 risposte, anonime, in
  `data/valutazione/risposte_questionario_anonime.csv`: è stata tolta solo l'ora di
  compilazione.
- **Annotazioni** — `data/annotazione_muscoli.csv` (125 nomi di esercizi annotati a mano,
  per misurare la precisione dell'abbinamento ai muscoli) e `data/valutazione/frasi_utente.csv`
  (72 frasi per confrontare regole e classificatori sul testo).

## Come si esegue

Serve Python 3.14 e una connessione a internet (per free-exercise-db e per il modello
degli embedding). La pipeline completa richiede circa 5 minuti.

```bash
python3.14 -m venv .venv
source .venv/bin/activate          # su Windows: .venv\Scripts\activate
pip install -r requirements.txt

# tutti gli script, in quest'ordine (12 va prima di 07, 22 prima di 17)
for n in 01 02 03 04 05 12 06 07 08 09 10 11 13 14 15 16 18 19 21 22 17; do
    python scripts/${n}_*.py
done

# la demo
streamlit run app_utente.py
```

## Struttura

```
archive/        dati grezzi da Kaggle (non inclusi)
src/            il codice del sistema: catalogo, punteggio, qualità, simulatore
scripts/        gli script della pipeline, numerati
data/           annotazioni a mano; processed/ è generata dalla pipeline
outputs/        tabelle e figure dei risultati
notebooks/      la baseline del notebook Kaggle
app_utente.py   la demo (Streamlit)
```

## Riferimenti

- Rosema, Torkamaan, Mansoury, *Personalized Recommender Systems for Gym Workouts: A
  Reinforcement Learning Approach*.
- Notebook Kaggle di partenza per la baseline:
  <https://www.kaggle.com/code/denvermagtibay/personalized-workout-program-recommender>

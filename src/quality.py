"""Indicatori di qualita' di un programma di allenamento.

Descrivono come e' costruito un programma, cosa che si vede solo dal dettaglio degli
esercizi. Ogni indicatore da' un valore grezzo e un voto fra 0 e 1; le soglie sono
tutte qui, cosi' si possono discutere, e 18_sensibilita.py le fa variare.
"""
import numpy as np
import pandas as pd

# --- gruppi muscolari ---
MACRO_GRUPPI = {
    "chest": "petto",
    "lats": "dorso", "middle back": "dorso", "lower back": "dorso", "traps": "dorso",
    "shoulders": "spalle",
    "biceps": "braccia", "triceps": "braccia", "forearms": "braccia",
    "quadriceps": "gambe", "hamstrings": "gambe", "glutes": "gambe",
    "calves": "gambe", "adductors": "gambe", "abductors": "gambe",
    "abdominals": "core", "neck": "core",
}

# Come dovrebbe dividersi il lavoro settimanale in un programma generalista.
# Valori presi dalla pratica, non dai dati.
RIFERIMENTO_EQUILIBRIO = {
    "gambe": 0.30, "dorso": 0.20, "petto": 0.15,
    "spalle": 0.13, "braccia": 0.14, "core": 0.08,
}

# --- soglie degli indicatori ---
# serie a settimana consigliate, per livello minimo dichiarato
VOLUME_CONSIGLIATO = {1: (25, 60), 2: (30, 75), 3: (40, 95), 4: (45, 120)}

FREQUENZA_IDEALE = 2.0
VARIETA_IDEALE = (0.05, 0.30)
PROGRESSIONE_PIENA = 0.10  # pendenza dell'RPE a settimana che vale progressione piena

FASCE_RIPETIZIONI = {
    "Powerlifting": (1, 6),
    "Olympic Weightlifting": (1, 6),
    "Powerbuilding": (3, 10),
    "Bodybuilding": (6, 15),
    "Muscle & Sculpting": (8, 20),
    "Bodyweight Fitness": (8, 25),
    "Athletics": (3, 12),
}

# Quanto in fretta cala il voto fuori dalla fascia, in frazioni della sua ampiezza.
# Sta qui e non dentro _plateau() perche' la sensibilita' deve poterlo cambiare.
# Sulla varieta' il lato basso non arriva mai a zero: la monotonia pesa poco.
TOLLERANZA_FASCIA = 0.5

# Pendenza dell'RPE a cui la progressione vale zero. Cosi' un RPE piatto, il caso
# piu' comune, prende anche la penalita' fissa. Non si ritocca, sarebbe tarare sui
# dati di valutazione.
PENDENZA_NULLA = 0.0

# Con meno di tre settimane la pendenza di una retta non dice niente.
SETTIMANE_MINIME_PROGRESSIONE = 3

PESI_QUALITA = {"volume": 0.20, "frequenza": 0.20, "progressione": 0.20,
                "equilibrio": 0.20, "varieta": 0.10, "coerenza": 0.10}
SOGLIA_COMPONENTE = 0.25
# Prima di confrontare un voto con la soglia lo si arrotonda: due programmi cadono
# esattamente sulla soglia, e l'errore di macchina li sposterebbe da una parte o
# dall'altra a seconda della versione di numpy.
DECIMALI_CONFRONTO = 12


def sotto_soglia(valori, soglia):
    """Dice dove il voto e' sotto la soglia, ignorando l'errore di macchina.

    I mancanti restano falsi, come nel confronto diretto.
    """
    return valori.round(DECIMALI_CONFRONTO) < soglia
PENALITA_COMPONENTE = 0.10


def _plateau(valore, minimo, massimo, tolleranza=None):
    """Dà 1 dentro la fascia consigliata e un voto che cala fuori.

    tolleranza: frazione dell'ampiezza oltre la quale il voto e' zero; se manca si usa
    TOLLERANZA_FASCIA.
    """
    tolleranza = TOLLERANZA_FASCIA if tolleranza is None else tolleranza
    valore = np.asarray(valore, dtype=float)
    ampiezza = (massimo - minimo) * tolleranza
    sotto = (valore - (minimo - ampiezza)) / ampiezza
    sopra = ((massimo + ampiezza) - valore) / ampiezza
    return np.clip(np.minimum(np.minimum(sotto, sopra), 1.0), 0, 1)


def quote_muscolari(dettaglio: pd.DataFrame, mappa: pd.DataFrame) -> pd.DataFrame:
    """Calcola la quota di serie per macrogruppo muscolare, una riga per programma.

    E' l'unica versione nel progetto: la usano equilibrio, catalogo e simulatore. Il
    dettaglio va passato gia' ripulito e ristretto al catalogo. I programmi senza
    esercizi abbinati non compaiono, perche' la loro distribuzione non si misura.
    """
    det = dettaglio.merge(
        mappa.loc[mappa["accettato"], ["exercise_name", "muscolo_primario"]],
        on="exercise_name", how="left")
    det["macro"] = det["muscolo_primario"].map(MACRO_GRUPPI)
    quote = (det.dropna(subset=["macro"]).groupby(["title", "macro"])["sets"].sum()
             .unstack(fill_value=0))
    for gruppo in RIFERIMENTO_EQUILIBRIO:
        if gruppo not in quote.columns:
            quote[gruppo] = 0.0
    quote = quote[list(RIFERIMENTO_EQUILIBRIO)]
    return quote.div(quote.sum(axis=1).replace(0, np.nan), axis=0)


def calcola_indicatori(dettaglio: pd.DataFrame, mappa: pd.DataFrame,
                       catalogo: pd.DataFrame) -> pd.DataFrame:
    """Calcola valori grezzi e voti degli indicatori, un record per programma."""
    det = dettaglio.merge(
        mappa.loc[mappa["accettato"], ["exercise_name", "muscolo_primario"]],
        on="exercise_name", how="left")
    det["macro"] = det["muscolo_primario"].map(MACRO_GRUPPI)

    info = catalogo.set_index("title")
    settimane = info["settimane"]

    ind = pd.DataFrame(index=info.index)

    # Volume: serie a settimana rispetto alla fascia del livello.
    ind["volume_settimanale"] = det.groupby("title")["sets"].sum() / settimane
    # Nel catalogo pulito il livello c'e' sempre; il fillna serve solo se la pulizia
    # cambiasse.
    livello = info["livello_min"].fillna(2).astype(int)
    minimi = livello.map(lambda l: VOLUME_CONSIGLIATO[l][0])
    massimi = livello.map(lambda l: VOLUME_CONSIGLIATO[l][1])
    ind["q_volume"] = [
        float(_plateau(v, mn, mx)) if np.isfinite(v) else np.nan
        for v, mn, mx in zip(ind["volume_settimanale"], minimi, massimi)]

    # Frequenza: quante volte a settimana si allena ogni gruppo.
    sessioni_macro = (det.dropna(subset=["macro"])
                      .groupby(["title", "macro"])
                      .apply(lambda d: d[["week", "day"]].drop_duplicates().shape[0],
                             include_groups=False))
    freq = (sessioni_macro / settimane.reindex(sessioni_macro.index.get_level_values(0)).values)
    # Media solo sui gruppi allenati: i gruppi assenti li punisce l'equilibrio,
    # non la frequenza.
    ind["frequenza_media"] = freq.groupby(level=0).mean()
    # Tagliata a 1 senza penalita' verso l'alto. Cosi' la frequenza distingue solo
    # i programmi scarsi, ma non diventa un secondo indicatore di volume.
    ind["q_frequenza"] = (ind["frequenza_media"] / FREQUENZA_IDEALE).clip(0, 1)

    # Progressione: pendenza dell'RPE, perche' il volume resta quasi sempre piatto.
    ind["pendenza_rpe"] = info["pendenza_rpe"]
    ind["q_progressione"] = ((ind["pendenza_rpe"] - PENDENZA_NULLA)
                             / (PROGRESSIONE_PIENA - PENDENZA_NULLA)).clip(0, 1)
    ind.loc[info["settimane"] < SETTIMANE_MINIME_PROGRESSIONE,
            "q_progressione"] = np.nan

    # Equilibrio: distanza dalla distribuzione di riferimento.
    # Prende il dettaglio originale perche' la funzione fa da se' il merge.
    quote = quote_muscolari(dettaglio, mappa)
    riferimento = pd.Series(RIFERIMENTO_EQUILIBRIO)
    distanza = (quote[riferimento.index] - riferimento).abs().sum(axis=1) / 2
    ind["distanza_equilibrio"] = distanza
    ind["q_equilibrio"] = (1 - distanza).clip(0, 1)

    # Varieta': poca e' monotonia, troppa e' dispersione.
    ind["varieta"] = info["varieta"]
    ind["q_varieta"] = [float(_plateau(v, *VARIETA_IDEALE)) for v in ind["varieta"]]

    # Coerenza: quota di serie nella fascia di ripetizioni dell'obiettivo.
    colonne_goal = {o: "goal_" + o.lower().replace(" & ", "_").replace(" ", "_")
                    for o in FASCE_RIPETIZIONI}
    fasce = {}
    for titolo, riga in info.iterrows():
        obiettivi = [o for o, c in colonne_goal.items() if riga.get(c, 0) == 1]
        if obiettivi:
            fasce[titolo] = (min(FASCE_RIPETIZIONI[o][0] for o in obiettivi),
                             max(FASCE_RIPETIZIONI[o][1] for o in obiettivi))
    # Con piu' obiettivi si prende l'involucro delle fasce, quindi piu' obiettivi
    # danno una fascia piu' larga e un voto piu' alto. Non lo correggo: la coerenza
    # abbassa l'NDCG, quindi non gonfia il guadagno.
    # reps > 0 esclude gli esercizi a tempo.
    a_ripetizioni = det[det["reps"] > 0].copy()
    a_ripetizioni["min_fascia"] = a_ripetizioni["title"].map(lambda t: fasce.get(t, (np.nan, np.nan))[0])
    a_ripetizioni["max_fascia"] = a_ripetizioni["title"].map(lambda t: fasce.get(t, (np.nan, np.nan))[1])
    # Senza obiettivo la fascia e' NaN e la coerenza verrebbe 0 invece di mancante.
    # Nel catalogo pulito non succede, perche' 02 toglie quei programmi.
    dentro = ((a_ripetizioni["reps"] >= a_ripetizioni["min_fascia"]) &
              (a_ripetizioni["reps"] <= a_ripetizioni["max_fascia"]))
    a_ripetizioni["dentro"] = dentro * a_ripetizioni["sets"]
    ind["q_coerenza"] = (a_ripetizioni.groupby("title")["dentro"].sum() /
                         a_ripetizioni.groupby("title")["sets"].sum())

    # --- qualita' complessiva ---
    componenti = ["q_volume", "q_frequenza", "q_progressione",
                  "q_equilibrio", "q_varieta", "q_coerenza"]
    pesi = pd.Series({f"q_{k}": v for k, v in PESI_QUALITA.items()})
    valori = ind[componenti]
    # il peso di una componente non misurabile passa alle altre
    pesi_validi = valori.notna() * pesi[componenti]
    somma = (valori.fillna(0) * pesi[componenti]).sum(axis=1) / pesi_validi.sum(axis=1)
    n_sotto = sotto_soglia(valori, SOGLIA_COMPONENTE).sum(axis=1)
    ind["qualita"] = (somma - PENALITA_COMPONENTE * n_sotto).clip(0, 1)
    ind["componenti_sotto_soglia"] = n_sotto
    return ind.reset_index().rename(columns={"index": "title"})


def aggrega_qualita(indicatori: pd.DataFrame, pesi: dict = None) -> pd.Series:
    """Ricalcola la qualita' complessiva con altri pesi.

    Serve all'ablation: con peso zero si misura quanto conta una componente.
    """
    pesi = pesi or PESI_QUALITA
    componenti = [f"q_{k}" for k in pesi]
    valori = indicatori[componenti]
    serie_pesi = pd.Series({f"q_{k}": v for k, v in pesi.items()})[componenti]
    pesi_validi = valori.notna() * serie_pesi
    totale = pesi_validi.sum(axis=1).replace(0, np.nan)
    somma = (valori.fillna(0) * serie_pesi).sum(axis=1) / totale
    attive = [c for c in componenti if serie_pesi[c] > 0]
    n_sotto = sotto_soglia(valori[attive], SOGLIA_COMPONENTE).sum(axis=1)
    return (somma - PENALITA_COMPONENTE * n_sotto).clip(0, 1)

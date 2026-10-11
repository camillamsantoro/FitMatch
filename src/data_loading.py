"""Caricamento e parsing di base del dataset dei programmi di allenamento."""
from pathlib import Path
import ast
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "archive"
SUMMARY_PATH = RAW_DIR / "program_summary.csv"
DETAIL_PATH = RAW_DIR / "programs_detailed_boostcamp_kaggle.csv"

# Impronte dei due CSV di Kaggle: se il file scaricato e' diverso, la lettura si
# ferma invece di cambiare i risultati senza nessun errore.
IMPRONTE_DATASET = {
    "program_summary.csv":
        "16f63d020d9d4f94ec3f247a4c43cec69a076af681672ba9a9b2b008f0817f28",
    "programs_detailed_boostcamp_kaggle.csv":
        "21e0980cc42bd420191cc3b9975f77b487dd55e159106b7f77a07819edf2da0f",
}
_verificati = set()

PROCESSED = ROOT / "data" / "processed"
OUT_FIG = ROOT / "outputs" / "figures"
OUT_TAB = ROOT / "outputs" / "tables"

DETAIL_COLS = ["title", "week", "day", "number_of_exercises",
               "exercise_name", "sets", "reps", "intensity"]
# Metadati che il dettaglio ripete su ogni riga; con piu' versioni sono quelli della
# versione della riga.
COLONNE_METADATI = ["description", "level", "goal", "equipment", "program_length",
                    "time_per_workout", "created", "last_edit"]


def verifica_impronta(percorso):
    """Controlla l'impronta SHA-256 di un CSV del dataset, una volta per esecuzione.

    Il dettaglio pesa quasi 300 MB, quindi il controllo non si ripete a ogni lettura."""
    percorso = Path(percorso)
    if percorso in _verificati:
        return
    import hashlib
    h = hashlib.sha256()
    with open(percorso, "rb") as f:
        for blocco in iter(lambda: f.read(1 << 20), b""):
            h.update(blocco)
    impronta = h.hexdigest()
    attesa = IMPRONTE_DATASET[percorso.name]
    if impronta != attesa:
        raise RuntimeError(
            f"{percorso} non e' il file del dataset usato dal progetto (SHA-256 "
            f"{impronta[:12]}..., atteso {attesa[:12]}...): scaricare di nuovo il dataset "
            f"da Kaggle e metterlo in archive/ senza modificarlo (README, sezione Dati).")
    _verificati.add(percorso)


def leggi_csv_dataset(percorso, **opzioni):
    """Legge un CSV del dataset dopo averne verificato l'impronta."""
    verifica_impronta(percorso)
    return pd.read_csv(percorso, **opzioni)


def parse_list(value):
    """Converte i campi level e goal, salvati come stringa "['Beginner', 'Novice']"."""
    if pd.isna(value):
        return []
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return [str(value).strip()]
    if isinstance(parsed, list):
        return [str(v).strip() for v in parsed]
    return [str(parsed).strip()]


def numera_versioni(df):
    """Separa le versioni: un programma e' una coppia (title, created).

    Se un titolo ha piu' date di creazione, ogni versione diventa "<titolo> (versione N)"
    in ordine di data; titolo_originale conserva il nome pubblicato.
    """
    date = (df[["title", "created"]].drop_duplicates()
            .sort_values(["title", "created"], kind="mergesort", na_position="last"))
    date["versione"] = date.groupby("title", dropna=False).cumcount() + 1
    date["versioni"] = date.groupby("title", dropna=False)["created"].transform("size")
    df = df.merge(date, on=["title", "created"], how="left")
    df["titolo_originale"] = df["title"]
    piu_versioni = df["versioni"] > 1
    df.loc[piu_versioni, "title"] = (df.loc[piu_versioni, "title"] + " (versione "
                                     + df.loc[piu_versioni, "versione"].astype(int).astype(str)
                                     + ")")
    return df.drop(columns=["versione", "versioni"])


def load_summary():
    """Restituisce un record per programma, con le versioni separate.

    program_summary.csv ha una riga sola per titolo, quindi ogni versione prende i
    metadati dalla prima delle sue righe del dettaglio, e total_exercises e' il numero
    delle sue righe.
    """
    prog = leggi_csv_dataset(SUMMARY_PATH)
    prog["titolo_originale"] = prog["title"]
    meta = numera_versioni(leggi_csv_dataset(DETAIL_PATH, usecols=["title"] + COLONNE_METADATI))
    meta = meta[meta["title"] != meta["titolo_originale"]]
    versioni = meta.groupby("title", sort=False).head(1).set_index("title")
    versioni["total_exercises"] = meta.groupby("title").size()
    versioni = versioni.reset_index()
    posizione = pd.Series(range(len(prog)), index=prog["title"])
    versioni["_posizione"] = versioni["titolo_originale"].map(posizione)
    prog["_posizione"] = range(len(prog))
    prog = prog[~prog["title"].isin(set(versioni["titolo_originale"]))]
    df = (pd.concat([prog, versioni[[c for c in prog.columns]]], ignore_index=True)
          .sort_values(["_posizione", "created"], kind="mergesort", na_position="last")
          .drop(columns="_posizione").reset_index(drop=True))
    df["level_list"] = df["level"].apply(parse_list)
    df["goal_list"] = df["goal"].apply(parse_list)
    return df


def load_detail(columns=None):
    """Legge il dettaglio con i titoli di numera_versioni().

    Oltre alle colonne chieste restituisce titolo_originale e created."""
    colonne = list(columns or DETAIL_COLS)
    lette = list(dict.fromkeys(colonne + ["title", "created"]))
    det = numera_versioni(leggi_csv_dataset(DETAIL_PATH, usecols=lette))
    return det[[c for c in det.columns if c in colonne]
               + [c for c in ("titolo_originale", "created") if c not in colonne]]


def split_reps(detail):
    """Separa ripetizioni e durate: nel dataset le ripetizioni negative sono secondi.

    Aggiunge reps_count (ripetizioni) e duration_s (secondi), NaN dove non servono.
    """
    df = detail.copy()
    is_time = df["reps"] < 0
    df["is_time_based"] = is_time
    df["reps_count"] = df["reps"].where(~is_time)
    df["duration_s"] = (-df["reps"]).where(is_time)
    return df

# Cambiare questa descrizione invalida la cache di testo_programmi.csv.
REGOLA_TESTO = "titolo + descrizione + 10 esercizi piu' frequenti, spareggio per nome"


def testo_programmi(catalogo):
    """Costruisce il testo di ogni programma per i sistemi testuali.

    Titolo, descrizione e i 10 esercizi piu' frequenti (a pari merito vince l'ordine
    alfabetico, cosi' il testo non cambia fra versioni di pandas). Il risultato va in
    cache e si rifa' se cambia la regola o mancano dei titoli.
    """
    percorso = PROCESSED / "testo_programmi.csv"
    testi = None
    if percorso.exists():
        cache = pd.read_csv(percorso)
        if ("regola" in cache.columns and (cache["regola"] == REGOLA_TESTO).all()
                and set(catalogo["title"]) <= set(cache["title"])):
            testi = cache.set_index("title")["testo"]
    if testi is None:
        det = load_detail(["title", "exercise_name"])
        conteggi = (det.groupby(["title", "exercise_name"]).size()
                    .rename("n").reset_index()
                    .sort_values(["title", "n", "exercise_name"],
                                 ascending=[True, False, True], kind="mergesort"))
        principali = (conteggi.groupby("title").head(10)
                      .groupby("title", sort=False)["exercise_name"].apply(" ".join))
        testi = (catalogo.set_index("title")["description"].fillna("") + " " +
                 catalogo.set_index("title").index.to_series() + " " +
                 principali.reindex(catalogo["title"]).fillna(""))
        percorso.parent.mkdir(parents=True, exist_ok=True)
        (testi.rename("testo").reset_index()
         .assign(regola=REGOLA_TESTO).to_csv(percorso, index=False))
    # str.lower di Python e non quello di pandas: pandas 3 tratta diversamente la
    # "İ" turca, e il testo (con la sua impronta) cambierebbe fra versioni.
    return [str(t).lower() for t in testi.reindex(catalogo["title"]).fillna("")]


def impronta_testi(testi) -> str:
    """Calcola l'impronta SHA-256 dei testi, nell'ordine in cui sono passati.

    Permette a 07 di verificare che gli embedding in cache vengano da questo testo."""
    import hashlib
    h = hashlib.sha256()
    for testo in testi:
        h.update(testo.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()

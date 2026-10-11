"""Lettura e formattazione della scheda di un programma."""
from functools import lru_cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCHEDE = ROOT / "data" / "processed" / "schede.csv.gz"


def schede_disponibili() -> bool:
    return SCHEDE.exists()


@lru_cache(maxsize=1)
def carica_tutte() -> pd.DataFrame:
    """Carica le schede una volta sola e le tiene in memoria."""
    return pd.read_csv(SCHEDE)


def carica_scheda(titolo: str) -> pd.DataFrame:
    tutte = carica_tutte()
    return tutte[tutte["title"] == titolo].copy()


def formatta_serie(riga) -> str:
    """Formatta una riga come '4 x 8 rip.' o '3 x 45 s', con l'RPE se c'e'."""
    serie = int(riga["sets"]) if pd.notna(riga["sets"]) else None
    if pd.notna(riga.get("duration_s")):
        quanto = f"{int(riga['duration_s'])} s"
    elif pd.notna(riga.get("reps_count")):
        quanto = f"{int(riga['reps_count'])} rip."
    else:
        quanto = None
    # Righe senza ripetizioni ne' durata: meglio dire cosa si sa che scrivere "1 x -".
    if quanto is None:
        testo = (f"{serie} serie" if serie else "") + " (ripetizioni non indicate)"
        testo = testo.strip()
    else:
        testo = f"{serie} x {quanto}" if serie else quanto
    if pd.notna(riga.get("intensity")) and riga["intensity"] > 0:
        testo += f"   RPE {riga['intensity']:.0f}"
    return testo


def giorni_della_settimana(scheda: pd.DataFrame, settimana: int) -> dict:
    """Restituisce un dizionario: numero del giorno -> tabella degli esercizi."""
    blocco_settimana = scheda[scheda["week"] == settimana]
    giorni = {}
    for giorno, blocco in blocco_settimana.groupby("day"):
        giorni[int(giorno)] = pd.DataFrame({
            "Esercizio": blocco["exercise_name"].values,
            "Serie": [formatta_serie(r) for _, r in blocco.iterrows()],
        })
    return giorni


def rpe_medio(righe: pd.DataFrame) -> float:
    """RPE medio delle sole righe che lo indicano.

    RPE 0 vuol dire "non indicato": contarlo come zero abbasserebbe la media."""
    return float(righe["intensity"].where(righe["intensity"] > 0).mean())

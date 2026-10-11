"""Costruzione del catalogo pulito dei programmi.

Ogni regola di pulizia e' esplicita e conta i casi che tocca.
"""
import numpy as np
import pandas as pd

from src.data_loading import load_summary, load_detail, split_reps
# La soglia sta in quality.py insieme alle altre, cosi' non ce ne sono due copie.
from src.quality import SETTIMANE_MINIME_PROGRESSIONE

LIVELLI = {"Beginner": 1, "Novice": 2, "Intermediate": 3, "Advanced": 4}
# Assunzione: chi ha piu' attrezzatura puo' fare i programmi che ne chiedono meno.
ATTREZZATURE = {"Dumbbell Only": 1, "At Home": 2, "Garage Gym": 3, "Full Gym": 4}
OBIETTIVI = ["Bodybuilding", "Muscle & Sculpting", "Powerbuilding", "Athletics",
             "Powerlifting", "Bodyweight Fitness", "Olympic Weightlifting"]

# Soglie di implausibilita', non giudizi sull'allenamento: stanno nella coda estrema
# dei dati e servono a scartare errori di inserimento.
MAX_GIORNI_SETTIMANA = 7  # fatto di calendario
MAX_SERIE_SETTIMANA = 200
MAX_ESERCIZI_SESSIONE = 20
MAX_DURATA_ESERCIZIO_S = 7200  # per UN esercizio
MAX_SERIE_PER_ESERCIZIO = 15


def indicatori_strutturali(detail):
    """Calcola gli indicatori strutturali dal dettaglio, un record per programma."""
    # La durata viene dal dettaglio e non dal campo dichiarato, perche' il dettaglio
    # descrive cio' che c'e' davvero. esercizi_sessione assume che ogni settimana abbia
    # tutti i giorni: e' un'approssimazione, ma non cambia nessuna esclusione.
    g = detail.groupby("title")
    feat = pd.DataFrame({
        "righe": g.size(),
        "settimane": g["week"].max(),
        "giorni_settimana": g["day"].max(),
        "esercizi_distinti": g["exercise_name"].nunique(),
        "serie_totali": g["sets"].sum(),
        "rpe_medio": g["intensity"].mean(),
        "rpe_max": g["intensity"].max(),
        # Attenzione: non e' un gruppo muscolare, anche se nel catalogo sta vicino
        # alle colonne quota_*. Non sommare tutte le colonne che iniziano per "quota_".
        "quota_a_tempo": g["is_time_based"].mean(),
    })
    feat["serie_settimana"] = feat["serie_totali"] / feat["settimane"]
    feat["esercizi_sessione"] = feat["righe"] / (feat["settimane"] * feat["giorni_settimana"])
    feat["varieta"] = feat["esercizi_distinti"] / feat["righe"]

    settimane = detail.groupby(["title", "week"]).agg(
        serie=("sets", "sum"), rpe=("intensity", "mean")).reset_index()

    def pendenza(dati, colonna):
        if len(dati) < SETTIMANE_MINIME_PROGRESSIONE:
            return np.nan
        return float(np.polyfit(dati["week"], dati[colonna], 1)[0])

    feat["pendenza_volume"] = settimane.groupby("title").apply(
        lambda d: pendenza(d, "serie"), include_groups=False)
    feat["pendenza_rpe"] = settimane.groupby("title").apply(
        lambda d: pendenza(d, "rpe"), include_groups=False)

    feat["settimane_identiche"] = detail.groupby("title").apply(
        lambda d: bool(d.groupby("week")["exercise_name"].apply(
            lambda s: tuple(sorted(s))).nunique() == 1 and d["week"].nunique() > 1),
        include_groups=False)
    return feat


def pulisci_dettaglio(detail):
    """Applica le correzioni sulle righe e restituisce (dataframe, conteggi)."""
    df = split_reps(detail)
    conteggi = {}

    durate_assurde = df["duration_s"] > MAX_DURATA_ESERCIZIO_S
    conteggi["righe con durata oltre 2 ore (azzerate)"] = int(durate_assurde.sum())
    df.loc[durate_assurde, "duration_s"] = np.nan

    serie_assurde = df["sets"] > MAX_SERIE_PER_ESERCIZIO
    conteggi[f"righe con piu' di {MAX_SERIE_PER_ESERCIZIO} serie (limitate)"] = int(serie_assurde.sum())
    df.loc[serie_assurde, "sets"] = MAX_SERIE_PER_ESERCIZIO

    return df, conteggi


def costruisci_catalogo():
    """Costruisce il catalogo pulito, un record per programma.

    Un programma e' una coppia (titolo, data di creazione): le versioni con lo stesso
    titolo sono programmi distinti (data_loading.numera_versioni).
    """
    prog = load_summary()
    detail, conteggi_righe = pulisci_dettaglio(load_detail())
    feat = indicatori_strutturali(detail)

    cat = prog.merge(feat, left_on="title", right_index=True, how="left")

    cat["livelli_num"] = cat["level_list"].apply(
        lambda vals: sorted(LIVELLI[v] for v in vals if v in LIVELLI))
    cat["livello_min"] = cat["livelli_num"].apply(lambda v: min(v) if v else np.nan)
    cat["livello_max"] = cat["livelli_num"].apply(lambda v: max(v) if v else np.nan)
    cat["attrezzatura_num"] = cat["equipment"].map(ATTREZZATURE)
    for obiettivo in OBIETTIVI:
        col = "goal_" + obiettivo.lower().replace(" & ", "_").replace(" ", "_")
        cat[col] = cat["goal_list"].apply(lambda vals, o=obiettivo: int(o in vals))

    esclusioni = {
        f"giorni a settimana oltre {MAX_GIORNI_SETTIMANA}": cat["giorni_settimana"] > MAX_GIORNI_SETTIMANA,
        f"serie a settimana oltre {MAX_SERIE_SETTIMANA}": cat["serie_settimana"] > MAX_SERIE_SETTIMANA,
        f"esercizi per sessione oltre {MAX_ESERCIZI_SESSIONE}": cat["esercizi_sessione"] > MAX_ESERCIZI_SESSIONE,
        "livello non dichiarato": cat["livelli_num"].apply(len) == 0,
        "obiettivo non dichiarato": cat["goal_list"].apply(len) == 0,
        "attrezzatura non dichiarata": cat["attrezzatura_num"].isna(),
        "durata del programma non dichiarata": cat["program_length"].isna(),
    }
    da_escludere = pd.Series(False, index=cat.index)
    conteggi_esclusioni = {}
    for motivo, maschera in esclusioni.items():
        maschera = maschera.fillna(False)
        conteggi_esclusioni[motivo] = int(maschera.sum())
        da_escludere |= maschera

    # i programmi segnalati restano nel catalogo
    cat["progressione_non_misurabile"] = (cat["settimane"]
                                          < SETTIMANE_MINIME_PROGRESSIONE)

    catalogo = cat.loc[~da_escludere].copy()
    versioni_per_titolo = prog["titolo_originale"].value_counts()
    report = {
        "titoli con piu' versioni": int((versioni_per_titolo > 1).sum()),
        "programmi in piu' grazie alla separazione":
            int(len(prog) - prog["titolo_originale"].nunique()),
        "righe corrette nel dettaglio": conteggi_righe,
        "programmi esclusi per motivo": conteggi_esclusioni,
        "programmi totali": len(cat),
        "programmi esclusi (unici)": int(da_escludere.sum()),
        "programmi nel catalogo pulito": len(catalogo),
        "di cui con progressione non misurabile": int(catalogo["progressione_non_misurabile"].sum()),
    }
    return catalogo, report

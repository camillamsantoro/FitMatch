"""Punteggio di rilevanza: quanto un programma e' adatto a un profilo.

L'attrezzatura e' l'unico vincolo rigido; gli altri requisiti diventano penalita'
graduali. Come in Rosema et al., una componente sotto soglia aggiunge una penalita'
fissa, cosi' un difetto grave non viene coperto dalla media.
"""
import numpy as np
import pandas as pd

from src.catalog import OBIETTIVI
from src.quality import RIFERIMENTO_EQUILIBRIO
from src.matching import ProfiloUtente

# Obiettivi vicini: un programma di powerbuilding va abbastanza bene a chi cerca
# powerlifting.
AFFINITA_OBIETTIVI = {
    ("Bodybuilding", "Muscle & Sculpting"): 0.85,
    ("Bodybuilding", "Powerbuilding"): 0.70,
    ("Muscle & Sculpting", "Powerbuilding"): 0.60,
    ("Powerlifting", "Powerbuilding"): 0.80,
    ("Powerlifting", "Olympic Weightlifting"): 0.50,
    ("Powerbuilding", "Olympic Weightlifting"): 0.40,
    ("Athletics", "Powerbuilding"): 0.50,
    ("Athletics", "Olympic Weightlifting"): 0.50,
    ("Athletics", "Bodyweight Fitness"): 0.45,
    ("Muscle & Sculpting", "Bodyweight Fitness"): 0.45,
    ("Bodybuilding", "Athletics"): 0.35,
}
AFFINITA_MINIMA = 0.15
# Il simulatore ha una costante con lo stesso nome ma un altro valore: e' voluto,
# perche' deve giudicare gli obiettivi in modo indipendente dal sistema. Non unificarle.

from src.quality import sotto_soglia as _sotto

PESI_DEFAULT = {"obiettivo": 0.32, "livello": 0.18, "tempo": 0.15,
                "giorni": 0.12, "sostenibilita": 0.13, "enfasi": 0.10}
SOGLIE_DEFAULT = {"obiettivo": 0.30, "livello": 0.40, "tempo": 0.35,
                  "giorni": 0.35, "sostenibilita": 0.35, "enfasi": 0.25}
# quanto lavoro in piu' sulle zone chieste serve per considerarle enfatizzate
FATTORE_ENFASI = 1.5
# pesi provati da 19 e 21, in un posto solo
PESI_ENFASI_PROVATI = [0.0, 0.06, 0.10, 0.15, 0.20]
PENALITA_SOGLIA = 0.15


def voto_sostenibilita(rapporto):
    """Dà un voto da 0 a 1 al rapporto fra volume del programma e capacita'.

    Pieno fra meta' capacita' e capacita' piena, poi cala da entrambi i lati. La usa
    anche il sistema adattivo.
    """
    rapporto = np.asarray(rapporto, dtype=float)
    return np.where(rapporto <= 1.0,
                    np.clip(rapporto / 0.5, 0, 1),
                    np.clip(1 - (rapporto - 1) / 0.8, 0, 1))


def _affinita(obiettivo_utente, obiettivo_programma):
    if obiettivo_utente == obiettivo_programma:
        return 1.0
    chiave = (obiettivo_utente, obiettivo_programma)
    return AFFINITA_OBIETTIVI.get(chiave,
           AFFINITA_OBIETTIVI.get(chiave[::-1], AFFINITA_MINIMA))


def fattibile(catalogo: pd.DataFrame, utente: ProfiloUtente) -> pd.Series:
    """Dice se l'attrezzatura del programma e' disponibile: e' l'unico vincolo rigido."""
    return catalogo["attrezzatura_num"] <= utente.attrezzatura_num


def componenti(catalogo: pd.DataFrame, utente: ProfiloUtente) -> pd.DataFrame:
    """Calcola le sei componenti di rilevanza, ognuna fra 0 e 1."""
    c = pd.DataFrame(index=catalogo.index)

    affinita_per_obiettivo = {o: _affinita(utente.obiettivo, o) for o in OBIETTIVI}
    valori = []
    for obiettivo in OBIETTIVI:
        col = "goal_" + obiettivo.lower().replace(" & ", "_").replace(" ", "_")
        valori.append(catalogo[col] * affinita_per_obiettivo[obiettivo])
    c["obiettivo"] = pd.concat(valori, axis=1).max(axis=1)

    distanza = np.maximum(catalogo["livello_min"] - utente.livello_num,
                          utente.livello_num - catalogo["livello_max"]).clip(lower=0)
    c["livello"] = (1 - 0.35 * distanza).clip(lower=0)

    # 15 minuti di sforamento tolgono meta' del voto
    sforo = (catalogo["time_per_workout"] - utente.minuti_per_seduta).clip(lower=0)
    c["tempo"] = 1 / (1 + sforo / 15)

    giorni_extra = (catalogo["giorni_settimana"] - utente.giorni_settimana).clip(lower=0)
    c["giorni"] = (1 - 0.40 * giorni_extra).clip(lower=0)

    # Sostenibilita': senza, premiare la qualita' porta a consigliare programmi
    # troppo pesanti per la persona.
    if "volume_settimanale" in catalogo.columns:
        rapporto = catalogo["volume_settimanale"] / max(utente.capacita_stimata, 1e-6)
        c["sostenibilita"] = voto_sostenibilita(rapporto)
    else:
        c["sostenibilita"] = 1.0

    # Enfasi: solo le zone dichiarate dalla persona, mai dedotte da altri dati.
    zone = [z for z in getattr(utente, "enfasi", ()) if f"quota_{z}" in catalogo.columns]
    if zone:
        # min_count=1 da' NaN se tutte le colonne mancano, invece di uno zero
        # silenzioso: cosi' i programmi senza muscoli abbinati si vedono.
        quota_richiesta = catalogo[[f"quota_{z}" for z in zone]].sum(axis=1, min_count=1)
        riferimento = sum(RIFERIMENTO_EQUILIBRIO[z] for z in zone)
        c["enfasi"] = (quota_richiesta / (riferimento * FATTORE_ENFASI)).clip(0, 1)
        # Per loro l'enfasi e' zero: stretching e corsa davvero non allenano le zone
        # chieste, e un NaN li toglierebbe dalla classifica.
        c["enfasi"] = c["enfasi"].fillna(0.0)
    else:
        c["enfasi"] = 1.0  # nessuna zona chiesta: la componente non incide
    return c


def rilevanza(catalogo: pd.DataFrame, utente: ProfiloUtente,
              pesi: dict = None, soglie: dict = None,
              dettaglio: bool = False):
    """Calcola il punteggio di rilevanza, fra 0 e 1."""
    # 'is None' e non 'or': un dizionario vuoto passato apposta verrebbe sostituito
    # dai valori predefiniti senza avviso.
    pesi = PESI_DEFAULT if pesi is None else pesi
    soglie = SOGLIE_DEFAULT if soglie is None else soglie
    c = componenti(catalogo, utente)

    somma = sum(c[k] * p for k, p in pesi.items()) / sum(pesi.values())
    # La penalita' vale solo per le componenti con peso > 0: chi "toglie" una
    # componente mettendo il peso a 0 deve toglierla davvero.
    sotto_soglia = sum(_sotto(c[k], s).astype(int) for k, s in soglie.items()
                       if pesi.get(k, 0) > 0)
    punteggio = (somma - PENALITA_SOGLIA * sotto_soglia).clip(lower=0, upper=1)
    punteggio = punteggio.where(fattibile(catalogo, utente), other=np.nan)

    if dettaglio:
        c = c.assign(componenti_sotto_soglia=sotto_soglia, rilevanza=punteggio)
        return c
    return punteggio


# --- punteggio finale: rilevanza + qualita' ---

LAMBDA_QUALITA = 0.30


def punteggio(catalogo: pd.DataFrame, utente: ProfiloUtente,
              lambda_qualita: float = LAMBDA_QUALITA,
              pesi: dict = None, soglie: dict = None) -> pd.Series:
    """Calcola il punteggio finale: rilevanza piu' qualita' pesata con lambda.

    Con lambda_qualita = 0 e' la sola rilevanza; il valore predefinito e' 0,3.
    """
    r = rilevanza(catalogo, utente, pesi=pesi, soglie=soglie)
    if "qualita" not in catalogo.columns:
        raise KeyError("il catalogo non contiene la colonna 'qualita': eseguire 05_features.py")
    # Dove la qualita' manca si usa la mediana. Nel sistema completo non succede;
    # nelle varianti dell'ablation con una sola componente si', e li' la mediana puo'
    # essere il voto peggiore.
    q = catalogo["qualita"].fillna(catalogo["qualita"].median())
    return (1 - lambda_qualita) * r + lambda_qualita * q

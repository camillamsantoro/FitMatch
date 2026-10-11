"""Simulatore di utenti.

Il dataset non ha utenti, quindi si valuta su utenti simulati, come in Rosema et al.
Ogni utente ha preferenze nascoste che il sistema non vede e un rumore personale, cosi'
il sistema non vince per costruzione. 06 misura quanto ogni fattore somiglia alla
componente del sistema che guarda la stessa cosa.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.catalog import LIVELLI, ATTREZZATURE, OBIETTIVI
from src.matching import ProfiloUtente
from src.quality import RIFERIMENTO_EQUILIBRIO

MACRO = list(RIFERIMENTO_EQUILIBRIO.keys())

# La popolazione non copia la composizione del catalogo: le persone vere non si
# distribuiscono come i programmi, ed e' cosi' che si vedono i buchi di copertura.
DISTRIBUZIONE_LIVELLO = {"Beginner": 0.30, "Novice": 0.25, "Intermediate": 0.30, "Advanced": 0.15}
DISTRIBUZIONE_ATTREZZATURA = {"Full Gym": 0.45, "Garage Gym": 0.20, "At Home": 0.25, "Dumbbell Only": 0.10}
DISTRIBUZIONE_OBIETTIVO = {"Bodybuilding": 0.25, "Muscle & Sculpting": 0.30, "Powerbuilding": 0.13,
                           "Powerlifting": 0.10, "Athletics": 0.10, "Bodyweight Fitness": 0.10,
                           "Olympic Weightlifting": 0.02}

CAPACITA_BASE = {1: 35, 2: 50, 3: 70, 4: 90}


@dataclass
class UtenteSimulato:
    # --- parte dichiarata: e' cio' che riceve il sistema ---
    obiettivo: str
    livello: str
    attrezzatura: str
    giorni_settimana: int
    minuti_per_seduta: int
    # --- parte nascosta: il sistema non la vede ---
    capacita_serie: float = 60.0
    distribuzione_desiderata: dict = field(default_factory=dict)
    varieta_preferita: float = 0.15
    rigidita_orario: float = 1.0
    seme: int = 0

    @property
    def enfasi_dichiarata(self) -> tuple:
        """Restituisce le zone che la persona indicherebbe nel modulo, al massimo due.

        Una persona non sa dire percentuali ma sa dire "gambe e glutei": si dichiarano
        solo le zone dove il desiderio supera chiaramente il riferimento.
        """
        if not self.distribuzione_desiderata:
            return ()
        scarti = {gruppo: quota - RIFERIMENTO_EQUILIBRIO[gruppo]
                  for gruppo, quota in self.distribuzione_desiderata.items()}
        marcate = sorted((g for g, scarto in scarti.items() if scarto >= 0.04),
                         key=lambda g: -scarti[g])
        return tuple(marcate[:2])

    def profilo_dichiarato(self) -> ProfiloUtente:
        return ProfiloUtente(self.obiettivo, self.livello, self.attrezzatura,
                             self.giorni_settimana, self.minuti_per_seduta,
                             enfasi=self.enfasi_dichiarata)


def _estrai(rng, distribuzione):
    valori = list(distribuzione)
    return str(rng.choice(valori, p=[distribuzione[v] for v in valori]))


def genera_utenti(n: int, seme: int = 42, scala_capacita: float = 1.0,
                  scala_rigidita: float = 1.0,
                  distribuzione_attrezzatura: dict = None,
                  scala_variabilita: float = 1.0) -> list:
    """Genera la popolazione simulata.

    I parametri opzionali servono alla prova di robustezza dello script 10.
    """
    rng = np.random.default_rng(seme)
    distribuzione_attrezzatura = distribuzione_attrezzatura or DISTRIBUZIONE_ATTREZZATURA
    utenti = []
    for i in range(n):
        livello = _estrai(rng, DISTRIBUZIONE_LIVELLO)
        livello_num = LIVELLI[livello]
        giorni = int(rng.choice([2, 3, 4, 5, 6], p=[0.10, 0.30, 0.30, 0.20, 0.10]))
        minuti = int(rng.choice([30, 45, 60, 75, 90], p=[0.20, 0.25, 0.30, 0.15, 0.10]))

        base = np.array([RIFERIMENTO_EQUILIBRIO[m] for m in MACRO])
        desiderata = np.clip(base + rng.normal(0, 0.05, len(MACRO)), 0.02, None)
        desiderata = desiderata / desiderata.sum()

        utenti.append(UtenteSimulato(
            obiettivo=_estrai(rng, DISTRIBUZIONE_OBIETTIVO),
            livello=livello,
            attrezzatura=_estrai(rng, distribuzione_attrezzatura),
            giorni_settimana=giorni,
            minuti_per_seduta=minuti,
            capacita_serie=float(CAPACITA_BASE[livello_num] * scala_capacita * (giorni / 4)
                                 * rng.normal(1.0, 0.18 * scala_variabilita)),
            distribuzione_desiderata=dict(zip(MACRO, desiderata)),
            varieta_preferita=float(np.clip(rng.normal(0.15, 0.07), 0.02, 0.5)),
            rigidita_orario=float(np.clip(rng.normal(1.0 * scala_rigidita, 0.4), 0.2, 3.5)),
            seme=int(rng.integers(0, 10**6)),
        ))
    return utenti


# --- utilita' nascosta ---

# Valori vicini a quelli del sistema ma non uguali: l'utente non ragiona con la
# tabella del progettista.
AFFINITA_PERCEPITA = {
    ("Bodybuilding", "Muscle & Sculpting"): 0.90,
    ("Bodybuilding", "Powerbuilding"): 0.65,
    ("Muscle & Sculpting", "Powerbuilding"): 0.55,
    ("Powerlifting", "Powerbuilding"): 0.85,
    ("Powerlifting", "Olympic Weightlifting"): 0.55,
    ("Athletics", "Bodyweight Fitness"): 0.55,
    ("Athletics", "Powerbuilding"): 0.45,
    ("Muscle & Sculpting", "Bodyweight Fitness"): 0.50,
}
# Diversa apposta da scoring.AFFINITA_MINIMA: non unificare.
AFFINITA_MINIMA = 0.10


def _affinita_percepita(a, b):
    if a == b:
        return 1.0
    return AFFINITA_PERCEPITA.get((a, b), AFFINITA_PERCEPITA.get((b, a), AFFINITA_MINIMA))


def fattori_utilita(catalogo: pd.DataFrame, quote_muscolari: pd.DataFrame,
                    utente: UtenteSimulato) -> dict:
    """Calcola i sette fattori dell'utilita', separati.

    Sono a parte perche' 06 li confronta uno per uno con le componenti del sistema,
    senza ricopiare questo codice.
    """

    affinita = {o: _affinita_percepita(utente.obiettivo, o) for o in OBIETTIVI}
    colonne = {o: "goal_" + o.lower().replace(" & ", "_").replace(" ", "_") for o in OBIETTIVI}
    f_obiettivo = pd.concat([catalogo[colonne[o]] * affinita[o] for o in OBIETTIVI],
                            axis=1).max(axis=1).clip(lower=AFFINITA_MINIMA)

    # livello: campana intorno alla fascia dichiarata
    livello_num = LIVELLI[utente.livello]
    distanza = np.maximum(catalogo["livello_min"] - livello_num,
                          livello_num - catalogo["livello_max"]).clip(lower=0)
    f_livello = np.exp(-0.5 * (distanza / 0.9) ** 2)

    # tempo: lo sforamento pesa di piu' per chi ha orari rigidi
    sforo = (catalogo["time_per_workout"] - utente.minuti_per_seduta).clip(lower=0)
    f_tempo = np.exp(-utente.rigidita_orario * sforo / 25)

    # giorni: con meno giorni di quelli richiesti il programma non si segue
    giorni_extra = (catalogo["giorni_settimana"] - utente.giorni_settimana).clip(lower=0)
    f_giorni = np.exp(-0.7 * giorni_extra)

    # carico: campana intorno alla capacita' personale, che il sistema non conosce
    rapporto = catalogo["volume_settimanale"] / utente.capacita_serie
    f_carico = np.exp(-0.5 * ((rapporto - 1) / 0.45) ** 2)

    desiderata = pd.Series(utente.distribuzione_desiderata)
    quote = quote_muscolari.reindex(catalogo["title"]).fillna(0)
    scarto = (quote[desiderata.index].values - desiderata.values).__abs__().sum(axis=1) / 2
    f_muscoli = pd.Series(np.exp(-2.0 * scarto), index=catalogo.index)

    f_varieta = np.exp(-0.5 * ((catalogo["varieta"] - utente.varieta_preferita) / 0.18) ** 2)

    return {"obiettivo": f_obiettivo, "livello": f_livello, "tempo": f_tempo,
            "giorni": f_giorni, "carico": f_carico, "muscoli": f_muscoli,
            "varieta": f_varieta}


def utilita(catalogo: pd.DataFrame, quote_muscolari: pd.DataFrame,
            utente: UtenteSimulato, con_rumore: bool = True,
            sigma_rumore: float = 0.12) -> pd.Series:
    """Calcola quanto l'utente apprezzerebbe ogni programma, fra 0 e 1.

    E' la verita' di riferimento. Uso la media geometrica: un fattore molto basso pesa
    tanto, ma l'utilita' non si concentra su pochi programmi come col prodotto. Il clip
    a 1e-6 fa si' che solo l'attrezzatura possa dare zero.
    """
    rng = np.random.default_rng(utente.seme)
    fattori = list(fattori_utilita(catalogo, quote_muscolari, utente).values())
    u = np.exp(np.mean([np.log(np.clip(f, 1e-6, None)) for f in fattori], axis=0))
    u = pd.Series(u, index=catalogo.index)

    if con_rumore:
        u = u * rng.lognormal(0, sigma_rumore, len(u))

    u = u.where(catalogo["attrezzatura_num"] <= ATTREZZATURE[utente.attrezzatura], 0.0)
    return (u / max(u.max(), 1e-9)).clip(0, 1)


# Oltre quante volte la capacita' un programma e' sovraccarico. Sta qui perche'
# 09 e 19 usino la stessa soglia.
SOGLIA_SOVRACCARICO = 1.2


def probabilita_rifiuto(catalogo: pd.DataFrame, utente: UtenteSimulato,
                        con_carico: bool = True) -> pd.Series:
    """Calcola la probabilita' che l'utente abbandoni il programma consigliato.

    Come il modello di rifiuto di Rosema et al., conta soprattutto il carico. I
    coefficienti sono scelti a mano. con_carico=False toglie il carico: 13 lo usa per
    mostrare che il sistema adattivo impara proprio da quello.
    """
    z = (-1.2
         + (1.8 if con_carico else 0.0)
         * (catalogo["volume_settimanale"] / utente.capacita_serie - 1).clip(lower=-1)
         + 0.9 * ((catalogo["time_per_workout"] - utente.minuti_per_seduta).clip(lower=0) / 30)
         + 0.8 * (catalogo["giorni_settimana"] - utente.giorni_settimana).clip(lower=0)
         + 0.6 * (catalogo["livello_min"] - LIVELLI[utente.livello]).clip(lower=0))
    return 1 / (1 + np.exp(-z))


def impronta_titoli(catalogo: pd.DataFrame) -> str:
    """Calcola l'impronta dell'elenco ordinato dei titoli.

    Cambia anche se il catalogo viene solo riordinato.
    """
    import hashlib
    return hashlib.sha256("\n".join(catalogo["title"].astype(str)).encode()).hexdigest()


def carica_verita(catalogo: pd.DataFrame, cartella) -> tuple:
    """Carica verita' di riferimento e utenti, controllando che corrispondano.

    Gli script rigenerano gli utenti dal seme invece di leggerli dal CSV: qui si
    controlla che le colonne di U siano i programmi giusti (con l'impronta dei titoli)
    e che tutti i campi salvati da 06 coincidano, non solo la capacita'.
    """
    from pathlib import Path
    cartella = Path(cartella)
    U = np.load(cartella / "utilita.npy")
    if U.shape[1] != len(catalogo):
        raise RuntimeError(f"utilita.npy ha {U.shape[1]} colonne ma il catalogo ha "
                           f"{len(catalogo)} programmi: rieseguire 06_simulatore.py.")
    file_impronta = cartella / "utilita_titoli.sha256"
    if not file_impronta.exists() or file_impronta.read_text().strip() != impronta_titoli(catalogo):
        raise RuntimeError("utilita.npy e' stata calcolata su un catalogo con titoli o "
                           "ordine diversi da quello attuale: rieseguire 06_simulatore.py.")
    utenti = genera_utenti(len(U))
    salvati = pd.read_csv(cartella / "utenti_simulati.csv", keep_default_na=False)
    problemi = []
    confronti = {
        "obiettivo": [u.obiettivo for u in utenti],
        "livello": [u.livello for u in utenti],
        "attrezzatura": [u.attrezzatura for u in utenti],
        "giorni": [u.giorni_settimana for u in utenti],
        "minuti": [u.minuti_per_seduta for u in utenti],
    }
    for campo, valori in confronti.items():
        if list(salvati[campo]) != list(valori):
            problemi.append(campo)
    for campo, attr, tolleranza in [("capacita_serie", "capacita_serie", 0.05),
                                    ("varieta_preferita", "varieta_preferita", 0.0005),
                                    ("rigidita_orario", "rigidita_orario", 0.005)]:
        rig = np.array([getattr(u, attr) for u in utenti])
        if not np.allclose(rig, salvati[campo].astype(float).values, rtol=0,
                           atol=tolleranza + 1e-9):
            problemi.append(campo)
    if "enfasi" in salvati.columns:
        rig = ["+".join(u.profilo_dichiarato().enfasi) for u in utenti]
        if list(salvati["enfasi"].astype(str)) != rig:
            problemi.append("enfasi")
    else:
        problemi.append("enfasi (colonna assente: rieseguire 06)")
    if problemi:
        raise RuntimeError("Gli utenti rigenerati non coincidono con quelli usati da "
                           "06_simulatore.py per calcolare utilita.npy. Campi diversi: "
                           + ", ".join(problemi) + ". Rieseguire 06 prima di questo script.")
    return U, utenti

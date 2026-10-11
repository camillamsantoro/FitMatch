"""Traduce i numeri del sistema in frasi per l'utente finale.

Ogni frase dice solo cio' che l'indicatore misura davvero: dire di piu' produce
frasi false.
"""
import pandas as pd

from src.catalog import OBIETTIVI
from src.quality import RIFERIMENTO_EQUILIBRIO
from src.scoring import FATTORE_ENFASI, _affinita, voto_sostenibilita
from src.simulator import SOGLIA_SOVRACCARICO


OBIETTIVI_IN_ITALIANO = {
    "Muscle & Sculpting": "tonificazione",
    "Bodybuilding": "aumento della massa muscolare",
    "Powerbuilding": "forza e massa",
    "Powerlifting": "forza sulle alzate",
    "Athletics": "preparazione atletica",
    "Bodyweight Fitness": "allenamento a corpo libero",
    "Olympic Weightlifting": "pesistica olimpica",
}


ZONE_IN_ITALIANO = {"gambe": "gambe e glutei", "dorso": "schiena", "petto": "petto",
                    "spalle": "spalle", "braccia": "braccia", "core": "addome"}

# Tre titoli sono sequenze di tasti o di cifre. Il programma resta in classifica,
# cambia solo come lo si presenta. Uso un elenco perche' le regole automatiche
# prendevano anche titoli legittimi.
TITOLI_ILLEGGIBILI = {"Xjxioxijxxkx", "fbgfgnfnfdn", "109273941442"}


def titolo_da_mostrare(titolo) -> str:
    titolo = str(titolo)
    if titolo.strip() in TITOLI_ILLEGGIBILI:
        return f"Programma senza un titolo leggibile (l'autore l'ha chiamato «{titolo}»)"
    return titolo


def _a(nomi: str) -> str:
    """Sceglie la preposizione davanti alla zona: "ad addome", "a schiena"."""
    return "ad " + nomi if nomi[0] in "aA" else "a " + nomi


def _zone(riga, profilo):
    return [z for z in getattr(profilo, "enfasi", ())
            if pd.notna(riga.get(f"quota_{z}"))]


def punto_enfasi(riga, profilo):
    """Scrive la frase sulle zone chieste, una zona alla volta.

    Si nomina solo una zona che il programma allena piu' del riferimento. "Molto
    lavoro" usa la stessa soglia della componente di enfasi.
    """
    molto, piu = [], []
    for z in _zone(riga, profilo):
        rapporto = riga[f"quota_{z}"] / RIFERIMENTO_EQUILIBRIO[z]
        if rapporto >= FATTORE_ENFASI:
            molto.append(ZONE_IN_ITALIANO[z])
        elif rapporto >= 1.05:
            piu.append(ZONE_IN_ITALIANO[z])
    if molto:
        return f"dedica molto lavoro {_a(' e '.join(molto))}, come hai chiesto"
    if piu:
        return f"dedica {_a(' e '.join(piu))} piu' lavoro di un programma equilibrato"
    return None


def punti_di_forza(riga) -> list:
    """Scrive le frasi positive a partire dagli indicatori di qualita'."""
    frasi = []
    # La progressione e' la pendenza dell'RPE: una tendenza, non un aumento a ogni
    # settimana e nemmeno il carico.
    if pd.notna(riga.get("q_progressione")) and riga["q_progressione"] >= 0.6:
        frasi.append("l'intensita' tende a crescere nel corso del programma")
    # Un gruppo a zero e' compatibile con un buon equilibrio, quindi "senza
    # trascurare intere zone" si dice solo se nessuna quota e' zero.
    if pd.notna(riga.get("q_equilibrio")) and riga["q_equilibrio"] >= 0.8:
        quote = [riga.get(f"quota_{z}") for z in RIFERIMENTO_EQUILIBRIO]
        nessuna_a_zero = all(pd.notna(q) and q > 0 for q in quote)
        frasi.append("distribuisce il lavoro fra i gruppi muscolari in modo equilibrato"
                     + (", senza trascurare intere zone" if nessuna_a_zero else ""))
    # e' una media: non dice niente del singolo gruppo
    if pd.notna(riga.get("q_frequenza")) and riga["q_frequenza"] >= 0.9:
        frasi.append("i muscoli allenati vengono ripresi in media piu' volte a settimana")
    if pd.notna(riga.get("q_varieta")) and riga["q_varieta"] >= 0.9:
        frasi.append("gli esercizi sono vari ma non dispersivi")
    # Guarda gli obiettivi del programma, che possono non essere quelli della persona.
    if pd.notna(riga.get("q_coerenza")) and riga["q_coerenza"] >= 0.85:
        frasi.append("serie e ripetizioni sono coerenti con l'obiettivo dichiarato dal "
                     "programma")
    return frasi


def _affinita_obiettivo(riga, profilo) -> float:
    """Calcola la migliore affinita' fra l'obiettivo della persona e quelli del programma."""
    valori = [_affinita(profilo.obiettivo, o)
              for o in OBIETTIVI
              if riga.get("goal_" + o.lower().replace(" & ", "_").replace(" ", "_"), 0) == 1]
    return max(valori, default=0.0)


def punti_di_attenzione(riga, profilo, capacita=None) -> list:
    """Elenca dove il programma si discosta dalla richiesta, dal piu' importante.

    Copre tutte le componenti graduali, altrimenti alcune schede direbbero "nessun
    compromesso" a torto. capacita: la stima aggiornata dopo gli scarti, se c'e'.
    """
    frasi = []
    if _affinita_obiettivo(riga, profilo) < 1:
        dichiarati = [OBIETTIVI_IN_ITALIANO[o] for o in OBIETTIVI
                      if riga.get("goal_" + o.lower().replace(" & ", "_").replace(" ", "_"), 0) == 1]
        frasi.append("e' pensato per un obiettivo diverso dal tuo"
                     + (f" ({', '.join(dichiarati)})" if dichiarati else ""))
    if pd.notna(riga.get("livello_min")) and riga["livello_min"] > profilo.livello_num:
        frasi.append("e' pensato per chi ha gia' un po' di esperienza in piu'")
    # Il voto di carico scende appena si esce dalla fascia, non solo oltre la soglia di
    # sovraccarico: avvisare piu' tardi lascerebbe schede con un compromesso nascosto.
    capacita = profilo.capacita_stimata if capacita is None else capacita
    volume = riga["volume_settimanale"]
    rapporto = volume / max(capacita, 1e-6)
    if rapporto > SOGLIA_SOVRACCARICO:
        frasi.append(f"prevede {volume:.0f} serie a settimana, piu' di quante ne stimo "
                     "adatte a te: all'inizio puoi toglierne qualcuna")
    elif rapporto > 1:
        frasi.append(f"prevede {volume:.0f} serie a settimana, un po' piu' di quante ne "
                     "stimo adatte a te")
    elif voto_sostenibilita(rapporto) < 1:
        frasi.append(f"con {volume:.0f} serie a settimana e' leggero rispetto a quello "
                     "che stimo tu possa reggere")
    minuti = riga["time_per_workout"]
    if minuti > profilo.minuti_per_seduta:
        frasi.append(f"ogni allenamento dura circa {minuti:.0f} minuti, "
                     f"{minuti - profilo.minuti_per_seduta:.0f} in piu' di quelli che hai indicato")
    giorni = riga["giorni_settimana"]
    if giorni > profilo.giorni_settimana:
        frasi.append(f"prevede {giorni:.0f} allenamenti a settimana invece di "
                     f"{profilo.giorni_settimana}")
    rapporti = {z: riga[f"quota_{z}"] / RIFERIMENTO_EQUILIBRIO[z] for z in _zone(riga, profilo)}
    sotto = [ZONE_IN_ITALIANO[z] for z, r in rapporti.items() if r < 1]
    pari = [ZONE_IN_ITALIANO[z] for z, r in rapporti.items() if 1 <= r < 1.05]
    if sotto:
        frasi.append(f"dedica {_a(' e '.join(sotto))} meno lavoro di un programma "
                     "equilibrato, anche se l'hai chiesto")
    if pari:
        frasi.append(f"dedica {_a(' e '.join(pari))} circa il lavoro di un programma "
                     "equilibrato, non di piu'")
    if pd.notna(riga.get("livello_max")) and riga["livello_max"] < profilo.livello_num:
        frasi.append("e' pensato per chi ha meno esperienza di te: potrebbe sembrarti facile")
    # Il voto e' tagliato a zero, quindi qui la pendenza puo' anche essere negativa:
    # si dice solo che manca una crescita pianificata.
    if pd.notna(riga.get("q_progressione")) and riga["q_progressione"] <= 0.05:
        frasi.append("l'intensita' non cresce nel corso del programma: gli aumenti di "
                     "carico dovrai gestirli tu")
    return frasi

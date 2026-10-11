"""Compatibilita' tra profilo utente e programma (parte knowledge-based)."""
from dataclasses import dataclass
import pandas as pd

from src.catalog import LIVELLI, ATTREZZATURE


@dataclass
class ProfiloUtente:
    obiettivo: str
    livello: str
    attrezzatura: str
    giorni_settimana: int
    minuti_per_seduta: int
    # testo libero della persona: lo usano solo le baseline testuali
    preferenze_testo: str = ""
    # zone dichiarate dalla persona, mai dedotte da altri dati
    enfasi: tuple = ()

    @property
    def livello_num(self):
        return LIVELLI[self.livello]

    @property
    def attrezzatura_num(self):
        return ATTREZZATURE[self.attrezzatura]

    @property
    def capacita_stimata(self) -> float:
        """Stima le serie a settimana sostenibili dai soli dati dichiarati.

        Una serie con il suo recupero dura circa 4,5 minuti, e il livello corregge la
        stima. Non c'e' un tetto: con molti giorni e minuti la stima supera la soglia
        di implausibilita' del catalogo, e li' la sostenibilita' premia i programmi pesanti.
        """
        serie_per_seduta = self.minuti_per_seduta / 4.5
        correzione_livello = {1: 0.80, 2: 0.90, 3: 1.00, 4: 1.10}[self.livello_num]
        return serie_per_seduta * self.giorni_settimana * correzione_livello


def compatibilita(catalogo: pd.DataFrame, utente: ProfiloUtente,
                  tolleranza_livello: int = 0,
                  tolleranza_minuti: int = 0,
                  tolleranza_giorni: int = 0,
                  richiedi_obiettivo: bool = True) -> pd.DataFrame:
    """Restituisce una maschera per vincolo e la colonna 'compatibile'.

    Le tolleranze rendono il filtro morbido, per quando quello rigido lascia troppi
    pochi programmi.
    """
    col_obiettivo = "goal_" + utente.obiettivo.lower().replace(" & ", "_").replace(" ", "_")

    m = pd.DataFrame(index=catalogo.index)
    m["attrezzatura"] = catalogo["attrezzatura_num"] <= utente.attrezzatura_num
    m["livello"] = (
        (catalogo["livello_min"] - tolleranza_livello <= utente.livello_num) &
        (utente.livello_num <= catalogo["livello_max"] + tolleranza_livello)
    )
    m["tempo"] = catalogo["time_per_workout"] <= utente.minuti_per_seduta + tolleranza_minuti
    # "tempo" usa la durata dichiarata dall'autore, "giorni" quella ricavata dal
    # dettaglio come max(day): dove la numerazione dei giorni ha buchi, max(day)
    # sovrastima.
    m["giorni"] = catalogo["giorni_settimana"] <= utente.giorni_settimana + tolleranza_giorni
    m["obiettivo"] = catalogo[col_obiettivo] == 1 if richiedi_obiettivo else True
    m["compatibile"] = m.all(axis=1)
    return m

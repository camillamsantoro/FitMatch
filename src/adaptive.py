"""Parte adattiva: il sistema impara dai rifiuti.

L'utente non conosce la propria capacita' di carico. Il sistema parte dalla stima
ricavata dai dati dichiarati e la abbassa quando l'utente rifiuta un programma pesante,
come nel "feedback di solo rifiuto" di Rosema et al. Le costanti sono scelte a mano.
"""
import numpy as np

from src.recommenders import NON_FATTIBILE
from src.scoring import voto_sostenibilita
from src.evaluation import ordina

# Un rifiuto dice qualcosa sul carico solo se il programma era vicino alla stima:
# un programma molto piu' leggero sara' stato rifiutato per altri motivi.
SOGLIA_RIFIUTO_PESANTE = 0.9
# Abbastanza da cambiare la proposta successiva, non da dimezzare la stima in due
# rifiuti.
PASSO_RIDUZIONE = 0.85
# Limiti di sicurezza per la demo, dove si puo' rifiutare quanto si vuole.
LIMITI = (0.3, 2.5)
# La correzione al massimo dimezza il punteggio: fa scendere un programma, non lo
# esclude.
PESO_CORREZIONE = 0.5


class StimaAdattiva:
    """Tiene un fattore correttivo sulla capacita' stimata dell'utente.

    Un'accettazione non alza la stima: la sessione finisce comunque alla prima
    accettazione.
    """

    def __init__(self, capacita_iniziale: float,
                 passo_riduzione: float = PASSO_RIDUZIONE, limiti=LIMITI):
        self.capacita_iniziale = capacita_iniziale
        self.fattore = 1.0
        self.passo_riduzione = passo_riduzione
        self.limiti = limiti
        self.storia = []

    @property
    def capacita(self) -> float:
        return self.capacita_iniziale * self.fattore

    def aggiorna(self, volume_programma: float, accettato: bool):
        """Abbassa la stima dopo il rifiuto di un programma pesante."""
        if not accettato and volume_programma > self.capacita * SOGLIA_RIFIUTO_PESANTE:
            self.fattore *= self.passo_riduzione
        self.fattore = float(np.clip(self.fattore, *self.limiti))
        self.storia.append(self.capacita)


def sessione(catalogo, profilo, utente, punteggi_base, probabilita_rifiuto_utente,
             adattivo: bool, n_proposte: int = 8, seme: int = 0,
             impara: bool = True):
    """Simula una sessione di proposte finche' l'utente non accetta.

    Con impara=False la correzione resta quella del primo giro: serve a separare la
    correzione dall'apprendimento. Restituisce il numero di proposte (None se nessuna
    accettata) e l'indice del programma accettato.
    """
    rng = np.random.default_rng(seme)
    stima = StimaAdattiva(profilo.capacita_stimata)
    volumi = catalogo["volume_settimanale"].values
    esclusi = set()

    for giro in range(n_proposte):
        punteggi = punteggi_base.copy()
        if adattivo:
            correzione = voto_sostenibilita(volumi / max(stima.capacita, 1e-6))
            punteggi = punteggi * ((1 - PESO_CORREZIONE) + PESO_CORREZIONE * correzione)
        if esclusi:
            punteggi[list(esclusi)] = NON_FATTIBILE
        proposto = int(ordina(punteggi, 1)[0])
        accettato = rng.random() > probabilita_rifiuto_utente[proposto]
        if adattivo and impara:
            stima.aggiorna(volumi[proposto], accettato)
        if accettato:
            return giro + 1, proposto
        esclusi.add(proposto)
    return None, None

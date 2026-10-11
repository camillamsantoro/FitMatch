"""I sistemi di raccomandazione messi a confronto.

Ognuno riceve il profilo dichiarato e da' un punteggio a ogni programma. Vanno dal
casuale al sistema completo, aggiungendo un pezzo alla volta. Tutti mandano in fondo
i programmi non eseguibili con l'attrezzatura della persona.
L'enfasi su una zona del corpo la vedono solo "rilevanza" e "completo".
"""
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.catalog import ATTREZZATURE, LIVELLI
from src.scoring import rilevanza, punteggio


NON_FATTIBILE = -np.inf


def _maschera_fattibili(catalogo, profilo):
    """Dice quali programmi si possono fare con l'attrezzatura dichiarata.

    E' la stessa regola di src.scoring.fattibile(): se cambia una, va cambiata l'altra.
    """
    return (catalogo["attrezzatura_num"] <= ATTREZZATURE[profilo.attrezzatura]).values


def _applica_fattibilita(punteggi, maschera):
    """Manda i non fattibili in fondo, qualunque sia la scala del sistema."""
    return np.where(maschera, np.asarray(punteggi, dtype=float), NON_FATTIBILE)


def casuale(catalogo, profilo, rng=None, **kwargs):
    # Il generatore arriva da fuori ed e' unico per tutta la valutazione: crearne
    # uno qui darebbe a tutti gli utenti la stessa classifica casuale.
    if rng is None:
        raise ValueError("casuale() richiede un rng creato una volta sola dal "
                         "chiamante: senza, tutti gli utenti riceverebbero la "
                         "stessa classifica.")
    p = rng.random(len(catalogo))
    return _applica_fattibilita(p, _maschera_fattibili(catalogo, profilo))


def solo_metadati(catalogo, profilo, **kwargs):
    """Conta quanti requisiti dichiarati il programma soddisfa.

    Non esclude niente, ordina: e' la versione piu' forte di un filtro rigido.
    """
    col = "goal_" + profilo.obiettivo.lower().replace(" & ", "_").replace(" ", "_")
    livello = LIVELLI[profilo.livello]
    soddisfatti = (
        (catalogo[col] == 1).astype(int)
        + ((catalogo["livello_min"] <= livello) & (livello <= catalogo["livello_max"])).astype(int)
        + (catalogo["time_per_workout"] <= profilo.minuti_per_seduta).astype(int)
        + (catalogo["giorni_settimana"] <= profilo.giorni_settimana).astype(int)
    ) / 4
    return _applica_fattibilita(soddisfatti.values,
                                _maschera_fattibili(catalogo, profilo))


class SistemaTestuale:
    """Misura la somiglianza TF-IDF fra il profilo e il testo del programma."""

    def __init__(self, catalogo, testi):
        self.vettorizzatore = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                              min_df=2, max_features=50_000)
        self.matrice = self.vettorizzatore.fit_transform(testi)

    def __call__(self, catalogo, profilo, **kwargs):
        # Stessa domanda degli altri sistemi testuali, da chiave_query, cosi'
        # ricevono tutti la stessa domanda.
        query = f"{chiave_query(profilo)} {profilo.preferenze_testo}"
        v = self.vettorizzatore.transform([query.lower()])
        somiglianza = cosine_similarity(v, self.matrice).ravel()
        return _applica_fattibilita(somiglianza,
                                    _maschera_fattibili(catalogo, profilo))


class SistemaSemantico:
    """Sistema testuale denso: TF-IDF ridotto con SVD (LSA).

    Riconosce anche parole diverse usate negli stessi contesti, cosa che il TF-IDF
    puro non fa.
    """

    def __init__(self, catalogo, testi, dimensioni=300, seme=0):
        self.vettorizzatore = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                              min_df=2, max_features=50_000)
        matrice = self.vettorizzatore.fit_transform(testi)
        self.svd = TruncatedSVD(n_components=dimensioni, random_state=seme)
        self.matrice = self.svd.fit_transform(matrice)

    def __call__(self, catalogo, profilo, **kwargs):
        # Stessa domanda degli altri sistemi testuali, da chiave_query, cosi'
        # ricevono tutti la stessa domanda.
        query = f"{chiave_query(profilo)} {profilo.preferenze_testo}"
        v = self.svd.transform(self.vettorizzatore.transform([query.lower()]))
        somiglianza = cosine_similarity(v, self.matrice).ravel()
        return _applica_fattibilita(somiglianza,
                                    _maschera_fattibili(catalogo, profilo))


class SistemaEmbeddings:
    """Sistema testuale con i sentence embeddings, come il notebook Kaggle.

    Usa i vettori calcolati da 12_embeddings.py; senza quel file il sistema non entra
    nel confronto.
    """

    def __init__(self, embeddings_programmi, embeddings_query):
        self.programmi = embeddings_programmi
        self.query = embeddings_query  # testo della query -> vettore

    def __call__(self, catalogo, profilo, **kwargs):
        chiave = chiave_query(profilo)
        v = self.query[chiave].reshape(1, -1)
        somiglianza = cosine_similarity(v, self.programmi).ravel()
        return _applica_fattibilita(somiglianza,
                                    _maschera_fattibili(catalogo, profilo))


def chiave_query(profilo):
    """Costruisce dal profilo la domanda dei sistemi testuali, in inglese.

    E' l'unica definizione, usata anche da 12_embeddings.py. L'enfasi non c'e', quindi
    persone con zone diverse possono ricevere la stessa classifica.
    """
    return (f"{profilo.obiettivo} {profilo.livello} {profilo.attrezzatura} "
            f"{profilo.giorni_settimana} day {profilo.minuti_per_seduta} minutes").lower()


def solo_rilevanza(catalogo, profilo, **kwargs):
    # Non fattibili a -inf come negli altri sistemi: 0 qui lo puo' prendere anche
    # un programma fattibile.
    return rilevanza(catalogo, profilo).fillna(NON_FATTIBILE).values


def completo(catalogo, profilo, lambda_qualita=0.30, **kwargs):
    return punteggio(catalogo, profilo,
                     lambda_qualita=lambda_qualita).fillna(NON_FATTIBILE).values


def costruisci_sistemi(catalogo, testi, embeddings=None):
    sistemi = {
        "casuale": casuale,
        "solo metadati": solo_metadati,
        "solo testo (TF-IDF)": SistemaTestuale(catalogo, testi),
        "solo testo (semantico LSA)": SistemaSemantico(catalogo, testi),
    }
    if embeddings is not None:
        sistemi["solo testo (embeddings)"] = SistemaEmbeddings(*embeddings)
    sistemi["rilevanza graduale"] = solo_rilevanza
    sistemi["completo (rilevanza + qualita)"] = completo
    return sistemi

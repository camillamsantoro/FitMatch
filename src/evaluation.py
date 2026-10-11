"""Metriche di valutazione sugli utenti simulati.

Tutte le classifiche passano da ordina(), che risolve i pari merito sempre allo stesso
modo. Per confrontare due sistemi si fa il bootstrap sulla differenza per utente, non
si confrontano due intervalli: gli utenti sono gli stessi.
"""
import numpy as np


SEME_PARI_MERITO = 12345
_permutazioni = {}


def _chiave_pari_merito(n):
    """Restituisce una permutazione fissa di 0..n-1, calcolata una volta sola."""
    if n not in _permutazioni:
        _permutazioni[n] = np.random.default_rng(SEME_PARI_MERITO).permutation(n)
    return _permutazioni[n]


def ordina(punteggi, k=None):
    """Ordina i programmi dal punteggio piu' alto al piu' basso.

    I pari merito si risolvono con una permutazione fissa. L'ordine interno di numpy non
    e' riproducibile, e l'ordine del catalogo farebbe consigliare a tutti la stessa
    decina quando i pari merito sono tanti, come nel sistema a metadati.
    """
    p = np.asarray(punteggi, dtype=float)
    ordine = np.lexsort((_chiave_pari_merito(len(p)), -p))
    return ordine[:k] if k is not None else ordine


def ndcg_at_k(punteggi, utilita, k=10):
    """Calcola l'NDCG con rilevanza graduata, cioe' l'utilita' vera dell'utente.

    Funziona anche per gli utenti con poca attrezzatura solo perche' utilita() mette a
    zero i programmi non fattibili prima di normalizzare."""
    ordine = ordina(punteggi, k)
    guadagni = utilita[ordine]
    sconti = 1 / np.log2(np.arange(2, len(guadagni) + 2))
    dcg = float((guadagni * sconti).sum())
    ideale = np.sort(utilita)[::-1][:k]
    idcg = float((ideale * sconti[:len(ideale)]).sum())
    return dcg / idcg if idcg > 0 else np.nan


def precision_at_k(punteggi, utilita, k=10, quota_rilevanti=0.05):
    """Calcola la quota di consigli fra i migliori per quell'utente.

    Rilevanti sono i migliori max(10, 5% dei fattibili): con pochi fattibili anche il
    caso prende molto, quindi non si confronta fra segmenti diversi."""
    fattibili = np.where(utilita > 0)[0]
    if len(fattibili) == 0:
        return np.nan
    n_rilevanti = max(k, int(len(fattibili) * quota_rilevanti))
    rilevanti = set(fattibili[ordina(utilita[fattibili], n_rilevanti)])
    consigliati = ordina(punteggi, k)
    return len(set(consigliati) & rilevanti) / k


def utilita_media_top_k(punteggi, utilita, k=10):
    return float(utilita[ordina(punteggi, k)].mean())


def copertura_catalogo(liste_consigliate, n_programmi):
    """Conta quanti programmi diversi il sistema propone almeno una volta."""
    return len(set().union(*liste_consigliate)) / n_programmi


def diversita_intra_lista(indici, quote_muscolari):
    """Misura quanto sono diversi i programmi consigliati come lavoro muscolare."""
    v = quote_muscolari[indici]
    if len(v) < 2:
        return np.nan
    distanze = [np.abs(v[i] - v[j]).sum() / 2
                for i in range(len(v)) for j in range(i + 1, len(v))]
    return float(np.mean(distanze))


def bootstrap_ic(valori, n=2000, alpha=0.05, seme=0):
    """Intervallo di confidenza al 95% sulla media, con ricampionamento degli utenti."""
    valori = np.asarray([v for v in valori if np.isfinite(v)])
    if len(valori) == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seme)
    medie = valori[rng.integers(0, len(valori), (n, len(valori)))].mean(axis=1)
    return (float(valori.mean()),
            float(np.percentile(medie, 100 * alpha / 2)),
            float(np.percentile(medie, 100 * (1 - alpha / 2))))


def bootstrap_ic_due_campioni(a, b, n=2000, alpha=0.05, seme=0):
    """Intervallo sulla differenza fra le medie di due gruppi diversi di utenti.

    Serve fra segmenti, dove gli utenti non sono gli stessi e il confronto appaiato non
    si puo' fare. Restituisce (media a - media b, inferiore, superiore).
    """
    a = np.asarray([v for v in a if np.isfinite(v)])
    b = np.asarray([v for v in b if np.isfinite(v)])
    if len(a) == 0 or len(b) == 0:
        raise ValueError("bootstrap_ic_due_campioni: uno dei due gruppi e' vuoto")
    rng = np.random.default_rng(seme)
    ma = a[rng.integers(0, len(a), (n, len(a)))].mean(axis=1)
    mb = b[rng.integers(0, len(b), (n, len(b)))].mean(axis=1)
    d = ma - mb
    return (float(a.mean() - b.mean()),
            float(np.percentile(d, 100 * alpha / 2)),
            float(np.percentile(d, 100 * (1 - alpha / 2))))


def intervallo_wilson(successi, prove, z=1.96):
    """Intervallo di Wilson al 95% per una proporzione.

    Con poche decine di casi l'approssimazione normale e' troppo stretta e puo' uscire
    da [0, 1].
    """
    import math
    if not prove:
        return (float("nan"), float("nan"))
    p = successi / prove
    d = 1 + z * z / prove
    centro = (p + z * z / (2 * prove)) / d
    meta = z * math.sqrt(p * (1 - p) / prove + z * z / (4 * prove * prove)) / d
    return (centro - meta, centro + meta)


# --- trasformazioni dei metadati (08 e 10) ---
# Quando un metadato prende il posto della qualita' in una somma pesata conta la forma
# della trasformazione, non solo l'ordine. 08 sceglie fra queste, 10 riusa la scelta.
def _minmax(x):
    x = np.asarray(x, dtype=float)
    return (x - x.min()) / (x.max() - x.min())


def _rango(x):
    from scipy.stats import rankdata
    return _minmax(rankdata(x))


def _log_minmax(x):
    return _minmax(np.log(np.asarray(x, dtype=float)))


TRASFORMAZIONI_METADATO = {
    "min-max": _minmax,
    "rango": _rango,
    "log + min-max": _log_minmax,
}


def trasforma_metadato(valori, trasformazione):
    """Applica una trasformazione per nome, come la scrive proxy_metadati.csv."""
    decrescente = trasformazione.endswith(", decrescente")
    nome = trasformazione.replace(", decrescente", "")
    colonna = TRASFORMAZIONI_METADATO[nome](valori)
    return 1 - colonna if decrescente else colonna

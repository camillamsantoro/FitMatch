"""Traduce la frase scritta dall'utente nel profilo che usa il sistema.

E' fatto a regole, senza modelli linguistici: e' piu' facile da spiegare e da
controllare, e cio' che non viene riconosciuto si corregge a mano.
"""
import re

# Alcune parole vengono dalle frasi di test, quindi la misura di 15 non e' fuori
# campione: la misura su lingua vera e' quella del questionario.
PAROLE_OBIETTIVO = {
    "Powerbuilding": ["powerbuilding", "forza e massa", "forte e grosso",
                      "forza e ipertrofia", "mix tra forza", "forza pura"],
    "Powerlifting": ["powerlifting", "massimale", "alzate", "panca squat stacco",
                     "forza massim", "gesti di gara", "sollevare piu' peso",
                     "sollevare piu peso"],
    "Olympic Weightlifting": ["pesistica", "olimpic", "strappo", "slancio",
                              "sollevamento olimpico"],
    "Bodyweight Fitness": ["corpo libero", "calisthenics", "senza attrezzi",
                           "a casa senza", "niente pesi", "nessun attrezzo",
                           "con il corpo", "peso corporeo"],
    "Athletics": ["atletic", "sport", "esplosiv", "performance", "prestazion",
                  "condizionamento", "reattiv", "scattante", "pallavolo", "calcio",
                  "basket", "rugby", "tennis", "maratona", "agonis"],
    "Muscle & Sculpting": ["tonific", "dimagr", "definizione", "definir", "snell",
                           "rassod", "in forma", "estetic", "asciug", "buttare giu",
                           "perdere peso", "perdere qualche chilo", "pancia"],
    "Bodybuilding": ["massa magra", "massa", "ipertrof", "muscol", "bodybuilding",
                     "crescere", "piu' grosso", "piu grosso"],
}

PAROLE_LIVELLO = {
    "Beginner": ["principiante", "mai allenat", "prima volta", "zero esperienza",
                 "mai messo piede", "mai fatto pesi", "da zero", "sedentari",
                 "non so da dove", "a digiuno"],
    "Novice": ["poca esperienza", "qualche mese", "alle prime armi", "da poco",
                "ho ripreso", "ho ricominciato", "sono tornato", "sono tornata",
                "riparto", "ho cominciato", "dopo un infortunio", "sei mesi fa",
                "due mesi fa"],
    "Intermediate": ["intermedio", "un anno", "un paio d'anni", "paio di stagioni",
                     "secondo anno", "dodici mesi", "tre anni", "amatoriale",
                     "abituat", "con costanza"],
    "Advanced": ["avanzat", "esperto", "anni di allenamento", "agonistic",
                 "spingo pesante", "diversi anni", "molti anni", "cinque anni",
                 "conosco bene gli esercizi", "da sempre"],
}

PAROLE_ATTREZZATURA = {
    "Dumbbell Only": ["solo manubri", "solo dei manubri", "soltanto manubri",
                      "due manubri", "un paio di manubri", "coppia di manubri",
                      "manubri e basta", "con i manubri", "manubri",
                      "pesetti", "kettlebell"],
    # Garage prima di At Home: "bilanciere a casa" contiene "a casa".
    "Garage Gym": ["garage", "home gym", "bilanciere a casa", "rack in casa"],
    "At Home": ["a casa", "in casa", "in camera", "in salotto", "corpo libero",
                "senza attrezzi", "nessun attrezzo", "niente attrezzi", "elastici",
                "mi arrangio", "poco spazio", "da casa",
                # senza questa regola "attrezzat" porterebbe a Full Gym
                "senza attrezzatura", "nessuna attrezzatura", "niente attrezzatura",
                "non ho attrezzatura"],
    "Full Gym": ["palestra", "sala pesi", "macchinari", "attrezzat",
                 "centro fitness", "centro attrezzat", "tutte le macchine",
                 "abbonamento"],
}
# Parole troppo generiche per battere un'altra attrezzatura nominata nella frase.
SENZA_PRECEDENZA = {"attrezzat", "abbonamento"}


def parole_palestra():
    return [p for p in PAROLE_ATTREZZATURA["Full Gym"] if p not in SENZA_PRECEDENZA]

# Parole brevi da cercare intere: "massa" sta dentro "massaggio". Le altre si cercano
# all'inizio di una parola, cosi' "tonific" prende anche "tonificare".
PAROLE_INTERE = {"massa", "lat", "core"}
# Negazioni dell'attrezzatura, l'unico vincolo rigido. Riconosco solo costruzioni
# precise ("non ho la palestra"): una regola piu' generica sbaglia in tutte e due le
# direzioni. L'elenco non basta, per questo sotto c'e' il dubbio.
_ART = r"(?:(?:la|una|il|un|i|le|lo|gli|l['’])\s*)?"
_NEGAZIONI_ATTREZZATURA = [
    rf"\bnon\s+(?:ho|possiedo|frequento|uso|ho\s+accesso\s+a)\s+(?:piu['’]?\s+|più\s+)?{_ART}{{P}}",
    rf"\bnon\s+vado\s+(?:piu['’]?\s+|più\s+)?(?:in|a|al|alla)\s+{_ART}{{P}}",
    rf"\bnon\s+mi\s+piace\s+{_ART}{{P}}",
    rf"\b(?:senza|niente|nessun[ao]?|zero|no)\s+{_ART}{{P}}",
    rf"\b(?:ho\s+)?(?:smesso|lasciat[oa]|mollat[oa])\s+(?:con\s+|di\s+andare\s+in\s+)?{_ART}{{P}}",
    rf"\b(?:odio|detesto)\s+{_ART}{{P}}",
    r"{P}\s*[?:]?\s*(?:no|mai)\b",  # "palestra? no" si', "palestra, no scuse" no
    r"{P}\s+(?:non\s+fa\s+per\s+me|l['’]ho\s+lasciata|l['’]ho\s+mollata)",
]
# Se una parola della palestra sta vicino a una negazione qualunque, il sistema non
# la usa e la demo chiede. Leggere palestra quando non c'e' propone programmi
# infattibili; non leggerla costa solo una domanda.
# "?" e ":" non sono confini: in "palestra? non ci vado" la negazione e' subito dopo.
_CONFINI = r"[.,;!\n()]|\b(?:ma|però|pero['’]|invece|mentre|anche se)\b"
_SEGNALI_NEGAZIONE = (r"\b(?:non|mai|senza|niente|nessun\w*|no|né|ne['’]|zero|smess\w*|"
                      r"smettere|lasciat\w*|mollat\w*|disdett\w*|odio|detesto)\b")

# Gli stessi limiti della demo, che li importa da qui.
GIORNI_MIN, GIORNI_MAX = 1, 7
MINUTI_MIN, MINUTI_MAX = 15, 120
# Valori di partenza del modulo della demo. L'attrezzatura manca apposta: la demo
# la chiede.
VALORI_INIZIALI = {"obiettivo": "Muscle & Sculpting", "livello": "Beginner",
                   "giorni_settimana": 3, "minuti_per_seduta": 45, "enfasi": ()}

_LETTERE = "a-zàèéìíòóù"


def _posizioni(testo: str, parola: str):
    """Trova dove 'parola' compare all'inizio di una parola del testo."""
    fine = rf"(?![{_LETTERE}])" if parola in PAROLE_INTERE else ""
    return [m.start() for m in re.finditer(rf"(?<![{_LETTERE}]){re.escape(parola)}{fine}", testo)]


def _negata(testo: str, posizione: int, lunghezza: int) -> bool:
    """Dice se la parola in 'posizione' sta dentro una negazione dell'elenco."""
    parola = re.escape(testo[posizione:posizione + lunghezza])
    for schema in _NEGAZIONI_ATTREZZATURA:
        for m in re.finditer(schema.replace("{P}", parola), testo):
            if m.start() <= posizione < m.end():
                return True
    return False


def _proposizione(testo: str, posizione: int) -> str:
    """Restituisce il pezzo di frase, fra due confini, che contiene 'posizione'."""
    inizio, fine = 0, len(testo)
    for m in re.finditer(_CONFINI, testo):
        if m.end() <= posizione:
            inizio = m.end()
        elif m.start() > posizione:
            fine = m.start()
            break
    return testo[inizio:fine]


def _in_dubbio(testo: str, posizione: int) -> bool:
    return re.search(_SEGNALI_NEGAZIONE, _proposizione(testo, posizione)) is not None


def _palestra(testo: str):
    """Divide le parole della palestra in usabili e messe da parte (negate o in dubbio)."""
    usabili, da_parte = [], []
    for parola in PAROLE_ATTREZZATURA["Full Gym"]:
        for i in _posizioni(testo, parola):
            dubbia = _negata(testo, i, len(parola)) or _in_dubbio(testo, i)
            (da_parte if dubbia else usabili).append(parola)
    return usabili, da_parte


def motivo_astensione(testo: str):
    """Dice perche' il sistema non ha letto la palestra: "conflitto", "dubbio" o None.

    Serve alla demo per fare la domanda giusta: a chi scrive "in palestra con i
    manubri" non si chiede se va in palestra.
    """
    t = testo.lower()
    if "attrezzatura" in interpreta(testo):
        return None
    usabili, da_parte = _palestra(t)
    if any(p in parole_palestra() for p in usabili):
        return "conflitto"
    if any(p in parole_palestra() for p in da_parte):
        return "dubbio"
    return None


def palestra_incerta(testo: str) -> bool:
    """Dice se la frase nomina la palestra ma il sistema si e' astenuto."""
    t = testo.lower()
    usabili, da_parte = _palestra(t)
    nominata = any(p in parole_palestra() for p in usabili + da_parte)
    return nominata and "attrezzatura" not in interpreta(testo)


def _trovata(testo: str, parola: str, nega: bool = False) -> bool:
    return any(not (nega and _negata(testo, i, len(parola))) for i in _posizioni(testo, parola))


def _cerca(testo: str, vocabolario: dict, nega: bool = False):
    """Restituisce il primo valore del vocabolario che ha una parola nella frase.

    Se ce ne sono piu' d'uno vince l'ordine del vocabolario, non quello della frase.
    """
    for valore, parole in vocabolario.items():
        for parola in parole:
            if _trovata(testo, parola, nega):
                return valore
    return None


_NUMERI = {"un": 1, "uno": 1, "una": 1, "due": 2, "tre": 3, "quattro": 4, "cinque": 5,
           "sei": 6, "sette": 7}
_NUMERI_MINUTI = {"dieci": 10, "quindici": 15, "venti": 20, "venticinque": 25,
                  "trenta": 30, "quaranta": 40, "quarantacinque": 45,
                  "cinquanta": 50, "sessanta": 60, "novanta": 90}
_N = r"(\d+|" + "|".join(sorted(_NUMERI, key=len, reverse=True)) + r")"


def _valore(token: str) -> int:
    return int(token) if token.isdigit() else _NUMERI[token]


def _giorni(t: str):
    """Legge gli allenamenti a settimana.

    "volte" e "sedute" sono sempre una frequenza, "giorni" non sempre: oltre 7 si
    ignora. In un intervallo vale il valore basso, quello che la persona ha di sicuro.
    """
    if re.search(r"\btutti i giorni\s+(tranne|meno|esclus)", t):
        return 6
    if re.search(r"\btutti i giorni\b", t):
        return 7
    for unita in [r"(volte|volta|allenamenti|sedute)", r"giorni"]:
        for m in re.finditer(rf"\b{_N}\s*(?:(?:-|–|/|o|a)\s*{_N}\s*)?{unita}\b", t):
            if re.match(r"\s*al giorno\b", t[m.end():]):
                continue
            v = _valore(m.group(1))
            if GIORNI_MIN <= v <= GIORNI_MAX:
                return v
    return None


def _minuti(t: str):
    """Legge i minuti per allenamento; in un intervallo vale il valore basso."""
    ore_e_mezza = re.search(rf"\b{_N}['’]?\s*or[ae]\s*e\s*mezz", t)
    if ore_e_mezza:
        return _valore(ore_e_mezza.group(1)) * 60 + 30
    if re.search(r"\btre quarti d['’]?\s*ora\b", t):
        return 45
    m = re.search(r"\b(\d+)\s*(?:(?:-|–|/|o|a)\s*\d+\s*)?(minuti|min\b|['’](?!\w))", t)
    if m:
        return int(m.group(1))
    h = re.search(r"\b(\d)\s*h\b", t)
    if h and int(h.group(1)) <= 3:
        return int(h.group(1)) * 60
    for parola, v in _NUMERI_MINUTI.items():
        if re.search(rf"\b{parola}\s*(minuti|min)\b", t):
            return v
    if re.search(r"\boretta\b", t):
        return 60
    if re.search(r"\bmezz['’e ]?\s*ora\b", t):
        return 30
    if re.search(r"\b(un|una|1)['’]?\s*ora\b", t):
        return 60
    ore = re.search(rf"\b{_N}\s*ore\b", t)
    # oltre tre ore non e' un allenamento ("dormire 8 ore")
    if ore and _valore(ore.group(1)) <= 3:
        return _valore(ore.group(1)) * 60
    return None


def interpreta(testo: str) -> dict:
    """Restituisce i campi riconosciuti; quelli assenti non si inventano."""
    t = testo.lower()
    risultato = {}

    obiettivo = _cerca(t, PAROLE_OBIETTIVO)
    if obiettivo:
        risultato["obiettivo"] = obiettivo
    livello = _cerca(t, PAROLE_LIVELLO)
    if livello:
        risultato["livello"] = livello
    # Le negazioni si controllano solo qui: leggere al contrario un vincolo rigido
    # fa proporre programmi che la persona non puo' fare.
    attrezzatura = _cerca(t, {k: v for k, v in PAROLE_ATTREZZATURA.items() if k != "Full Gym"},
                          nega=True)
    # Palestra e un'altra attrezzatura insieme puo' essere "in palestra con i
    # manubri" ma anche "andavo in palestra, ora mi alleno a casa". Nel dubbio il
    # sistema si astiene e la demo chiede.
    usabili, _ = _palestra(t)
    forti = [p for p in usabili if p in parole_palestra()]
    if forti and attrezzatura:
        attrezzatura = None
    elif forti or (usabili and not attrezzatura):
        attrezzatura = "Full Gym"
    if attrezzatura:
        risultato["attrezzatura"] = attrezzatura

    giorni = _giorni(t)
    if giorni is not None:
        risultato["giorni_settimana"] = giorni
    minuti = _minuti(t)
    if minuti is not None:
        risultato["minuti_per_seduta"] = max(MINUTI_MIN, min(MINUTI_MAX, minuti))
    return risultato


PAROLE_ENFASI = {
    "gambe": ["gambe", "glutei", "quadricipiti", "femorali", "polpacci", "cosce",
              "parte bassa", "lower body"],
    "dorso": ["schiena", "dorsali", "dorso", "trapezi", "lat"],
    "petto": ["petto", "pettorali", "panca"],
    "spalle": ["spalle", "deltoidi"],
    "braccia": ["braccia", "bicipiti", "tricipiti", "avambracci"],
    "core": ["addome", "addominali", "core", "pancia piatta", "cintura addominale"],
}


# Controlla che le zone siano gli stessi sei gruppi usati altrove, invece di
# fidarsi di tre elenchi scritti a mano.
from src.quality import RIFERIMENTO_EQUILIBRIO
assert set(PAROLE_ENFASI) == set(RIFERIMENTO_EQUILIBRIO), "zone dell'enfasi diverse dai gruppi"


def zone_indicate(testo: str) -> tuple:
    """Trova le zone del corpo nominate nella frase, al massimo due, in ordine."""
    t = testo.lower()
    trovate = []
    for zona, parole in PAROLE_ENFASI.items():
        posizioni = [i for p in parole for i in _posizioni(t, p)]
        if posizioni:
            trovate.append((min(posizioni), zona))
    return tuple(zona for _, zona in sorted(trovate))[:2]


ETICHETTE = {
    "obiettivo": "obiettivo", "livello": "livello", "attrezzatura": "attrezzatura",
    "giorni_settimana": "giorni a settimana", "minuti_per_seduta": "minuti per allenamento",
    "enfasi": "zone su cui concentrarsi",
}

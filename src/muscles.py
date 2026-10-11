"""Abbinamento degli esercizi del dataset ai gruppi muscolari.

Il dataset ha solo il nome dell'esercizio, non i muscoli: i nomi si abbinano a quelli
di un database aperto (free-exercise-db). Per ogni coppia si salva la somiglianza,
cosi' la copertura e' misurabile.
"""
import json
import re
import urllib.request
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "processed" / "free_exercise_db.json"
# Il database si scarica da un commit fissato, non dal ramo main che puo' cambiare;
# l'impronta verifica che il file sia quello del progetto.
DB_COMMIT = "f00c92c7dcf1216a928a52c3706c7ce8e2f71ed5"
DB_URL = ("https://raw.githubusercontent.com/yuhonas/free-exercise-db/"
          f"{DB_COMMIT}/dist/exercises.json")
DB_SHA256 = "5bb747e3fc658f095a60dcbf6d53c96627acdcc6ffb6fffde86f7e26995d40bf"

RUMORE = {"machine", "smith", "assisted", "weighted", "single", "arm", "leg",
          "alternating", "standing", "seated", "lying", "bar", "grip", "wide",
          "close", "neutral", "reverse", "front", "rear", "side", "high", "low",
          "and", "with", "the", "of", "to"}

SOGLIA_ACCETTAZIONE = 0.60


def scarica_database():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not DB_PATH.exists():
        urllib.request.urlretrieve(DB_URL, DB_PATH)
    import hashlib
    impronta = hashlib.sha256(DB_PATH.read_bytes()).hexdigest()
    if impronta != DB_SHA256:
        raise RuntimeError(f"{DB_PATH.name} non e' quello del progetto (SHA-256 {impronta[:12]}..., "
                           f"atteso {DB_SHA256[:12]}...): cancellarlo e rieseguire 04_muscoli.py, "
                           f"che lo riscarica dal commit {DB_COMMIT[:12]}.")
    with open(DB_PATH) as f:
        return json.load(f)


def normalizza(nome: str) -> set:
    nome = nome.lower()
    nome = re.sub(r"[^a-z0-9\s]", " ", nome)
    token = {t for t in nome.split() if len(t) > 2 and t not in RUMORE}
    return token


def somiglianza(token_a: set, token_b: set, nome_a: str, nome_b: str) -> float:
    """Calcola la somiglianza fra due nomi: parole in comune piu' somiglianza letterale."""
    if not token_a or not token_b:
        return 0.0
    jaccard = len(token_a & token_b) / len(token_a | token_b)
    copertura = len(token_a & token_b) / min(len(token_a), len(token_b))
    letterale = SequenceMatcher(None, nome_a.lower(), nome_b.lower()).ratio()
    return 0.4 * jaccard + 0.4 * copertura + 0.2 * letterale


# Regole manuali per esercizi frequenti che l'abbinamento automatico sbaglia o non
# trova. Sono poche ma coprono molte righe.
REGOLE_MANUALI = [
    (r"pec deck|chest fly|chest press|pectoral", "chest", "push", "isolation"),
    (r"reverse pec deck|rear delt|face pull", "shoulders", "pull", "isolation"),
    (r"tricep|skull ?crusher|pushdown|push down|overhead extension", "triceps", "push", "isolation"),
    (r"bicep|curl", "biceps", "pull", "isolation"),
    # subito dopo: i curl per le gambe non sono curl per i bicipiti
    (r"leg curl|hamstring curl|nordic|lying curl|seated leg curl|glute ham", "hamstrings", "pull", "isolation"),
    (r"chest supported row|t.?bar row|seal row|pendlay", "middle back", "pull", "compound"),
    (r"hip abduct|abductor", "abductors", "push", "isolation"),
    (r"hip adduct|adductor", "adductors", "pull", "isolation"),
    (r"hip thrust|glute bridge|glute kickback|glute", "glutes", "push", "compound"),
    (r"lateral raise|side raise|delt raise", "shoulders", "push", "isolation"),
    (r"calf", "calves", "push", "isolation"),
    (r"crunch|sit.?up|plank|ab wheel|leg raise|hanging knee", "abdominals", "static", "isolation"),

    # --- correzioni nate dalla verifica di precisione ---
    # Quando il nome di riferimento ha una sola parola generica ("press", "pull"),
    # l'abbinamento automatico supera la soglia per sbaglio: qui si fissa il muscolo
    # dell'esercizio canonico.
    (r"\bpull.?ups?\b|\bpullups?\b|\bchin.?ups?\b|\bpull.?down", "lats", "pull", "compound"),
    (r"overhead press|shoulder press|military press|arnold press|landmine press|\bohp\b", "shoulders", "push", "compound"),
    (r"\brow\b", "middle back", "pull", "compound"),
    # l'upright row tira verso l'alto: lavora le spalle, non il dorso
    (r"upright row", "shoulders", "pull", "compound"),
    (r"back extension|hyper ?extension|good.?morning", "lower back", "pull", "compound"),
    (r"rack pull|block pull", "lower back", "pull", "compound"),
    (r"french press", "triceps", "push", "isolation"),
    (r"bench press|larsen press|floor press", "chest", "push", "compound"),
    (r"wall sit", "quadriceps", "static", "isolation"),
    # i wrist curl lavorano gli avambracci, non i bicipiti
    (r"\bwrist\b", "forearms", "pull", "isolation"),
    # esercizi frequenti che l'abbinamento automatico non trova
    (r"pec fly|pec minor fly", "chest", "push", "isolation"),
    (r"kettlebell swing", "hamstrings", "pull", "compound"),
    (r"\bshrug", "traps", "pull", "isolation"),
    (r"dead hang", "lats", "static", "isolation"),

    # confini di parola: senza, "crunch" contiene "run" e verrebbe escluso
    (r"\b(?:stretch(?:ing)?|mobility|foam roll|warm.?up|cardio|walk(?:ing)?|run(?:ning)?|jog|bike|cycling|treadmill|erg|skierg)\b", None, None, None),

    # --- rientri dopo l'esclusione: l'ordine conta ---
    # La regola sopra esclude anche gli affondi camminati e il farmer's walk perche'
    # contengono "walk"; queste due regole li rimettono dentro, ma non gli
    # allungamenti in affondo.
    (r"\blunge(?!.*stretch)", "quadriceps", "push", "compound"),
    (r"farmer.?s?\s*walk", "forearms", "static", "compound"),
    (r"lateral.{0,10}walk|monster walk|band(?:ed)?.{0,8}walk", "abductors", "push", "isolation"),
]


def applica_regole_manuali(mappa):
    """Sostituisce gli abbinamenti automatici dove c'e' una regola manuale."""
    for pattern, muscolo, forza, meccanica in REGOLE_MANUALI:
        maschera = mappa["exercise_name"].str.lower().str.contains(pattern, regex=True, na=False)
        if muscolo is None:
            # non e' lavoro muscolare: escluso dall'equilibrio
            mappa.loc[maschera, ["muscolo_primario", "tipo_forza", "meccanica"]] = [None, None, None]
            mappa.loc[maschera, "accettato"] = False
            mappa.loc[maschera, "origine"] = "regola manuale (escluso)"
        else:
            mappa.loc[maschera, "muscolo_primario"] = muscolo
            mappa.loc[maschera, "tipo_forza"] = forza
            mappa.loc[maschera, "meccanica"] = meccanica
            mappa.loc[maschera, "accettato"] = True
            mappa.loc[maschera, "origine"] = "regola manuale"
    return mappa


def costruisci_mappatura(nomi_dataset) -> pd.DataFrame:
    db = scarica_database()
    riferimenti = [{
        "nome_db": e["name"],
        "token": normalizza(e["name"]),
        "muscoli_primari": e.get("primaryMuscles", []),
        "muscoli_secondari": e.get("secondaryMuscles", []),
        "tipo_forza": e.get("force"),
        "meccanica": e.get("mechanic"),
    } for e in db]

    righe = []
    for nome in nomi_dataset:
        token = normalizza(nome)
        migliore, punteggio_migliore = None, 0.0
        for rif in riferimenti:
            p = somiglianza(token, rif["token"], nome, rif["nome_db"])
            if p > punteggio_migliore:
                migliore, punteggio_migliore = rif, p
        righe.append({
            "exercise_name": nome,
            "nome_db": migliore["nome_db"] if migliore else None,
            "punteggio": round(punteggio_migliore, 3),
            "accettato": punteggio_migliore >= SOGLIA_ACCETTAZIONE,
            "muscolo_primario": (migliore["muscoli_primari"] or [None])[0] if migliore else None,
            "muscoli_primari": "|".join(migliore["muscoli_primari"]) if migliore else "",
            "tipo_forza": migliore["tipo_forza"] if migliore else None,
            "meccanica": migliore["meccanica"] if migliore else None,
        })
    mappa = pd.DataFrame(righe)
    mappa["origine"] = mappa["accettato"].map({True: "abbinamento automatico", False: "non abbinato"})
    return applica_regole_manuali(mappa)

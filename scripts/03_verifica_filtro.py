"""03 - Quanti programmi restano col filtro rigido, per profili tipici.

Se ne restano pochi, il filtro va reso morbido. Salva outputs/tables/verifica_filtro.csv
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.matching import ProfiloUtente, compatibilita
from src.data_loading import OUT_TAB

CATALOGO = ROOT / "data" / "processed" / "catalogo_pulito.csv"

PROFILI = {
    "principiante a casa":      ProfiloUtente("Muscle & Sculpting", "Beginner", "At Home", 3, 30),
    "principiante in palestra": ProfiloUtente("Bodybuilding", "Beginner", "Full Gym", 3, 60),
    "solo manubri":             ProfiloUtente("Muscle & Sculpting", "Novice", "Dumbbell Only", 3, 45),
    "a corpo libero":           ProfiloUtente("Bodyweight Fitness", "Beginner", "At Home", 4, 30),
    "intermedio in palestra":   ProfiloUtente("Bodybuilding", "Intermediate", "Full Gym", 4, 60),
    "intermedio garage gym":    ProfiloUtente("Powerbuilding", "Intermediate", "Garage Gym", 4, 75),
    "avanzato powerlifting":    ProfiloUtente("Powerlifting", "Advanced", "Full Gym", 5, 90),
    "avanzato pesistica":       ProfiloUtente("Olympic Weightlifting", "Advanced", "Full Gym", 5, 90),
}

# Il livello puo' sbagliare in tutte e due le direzioni; tempo e giorni solo verso
# l'alto, perche' un programma piu' corto va sempre bene. I 15 minuti sono gli stessi
# di src/scoring.py.
SCENARI = {
    "rigido":      dict(tolleranza_livello=0, tolleranza_minuti=0, tolleranza_giorni=0, richiedi_obiettivo=True),
    "morbido":     dict(tolleranza_livello=1, tolleranza_minuti=15, tolleranza_giorni=1, richiedi_obiettivo=True),
    "senza obiettivo": dict(tolleranza_livello=1, tolleranza_minuti=15, tolleranza_giorni=1, richiedi_obiettivo=False),
}


def main():
    catalogo = pd.read_csv(CATALOGO)
    print(f"Catalogo pulito: {len(catalogo)} programmi\n")

    righe = []
    for nome, utente in PROFILI.items():
        riga = {"profilo": nome}
        for scenario, parametri in SCENARI.items():
            m = compatibilita(catalogo, utente, **parametri)
            riga[scenario] = int(m["compatibile"].sum())
        # Ogni vincolo preso da solo: confrontato con lo scenario rigido mostra che il
        # problema e' l'intersezione, non un vincolo in particolare.
        m = compatibilita(catalogo, utente, **SCENARI["rigido"])
        singoli = {c: int(m[c].sum())
                   for c in ["attrezzatura", "livello", "tempo", "giorni", "obiettivo"]}
        riga["vincolo piu' selettivo"] = min(singoli, key=singoli.get)
        riga["programmi che lo superano"] = singoli[riga["vincolo piu' selettivo"]]
        righe.append(riga)

    tab = pd.DataFrame(righe).set_index("profilo")
    print(tab.to_string())

    # Il doppio della top-10: sotto, ordinare non serve piu' a niente.
    print("\nSoglia di guardia: sotto i 20 programmi compatibili l'ordinamento perde senso")
    print("(20 = il doppio della top-10 su cui si valuta).")
    critici = tab[tab["rigido"] < 20].index.tolist()
    print("Profili critici con filtro rigido:", critici if critici else "nessuno")
    critici_morbidi = tab[tab["morbido"] < 20].index.tolist()
    print("Profili ancora critici con filtro morbido:", critici_morbidi if critici_morbidi else "nessuno")
    tab.to_csv(OUT_TAB / "verifica_filtro.csv")


if __name__ == "__main__":
    main()

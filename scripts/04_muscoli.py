"""04 - Abbinamento esercizi -> gruppi muscolari e misura della copertura.

Salva data/processed/mappatura_esercizi.csv e outputs/tables/copertura_muscoli.csv
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loading import load_detail, OUT_TAB
from src.muscles import costruisci_mappatura, SOGLIA_ACCETTAZIONE
from src.quality import MACRO_GRUPPI
from src.evaluation import intervallo_wilson

PROCESSED = ROOT / "data" / "processed"


def main():
    # La copertura si misura sul dettaglio grezzo, anche sui programmi esclusi dopo:
    # misura la mappatura come strumento, non il catalogo.
    det = load_detail(["title", "exercise_name"])
    frequenze = det["exercise_name"].value_counts()
    print(f"Esercizi distinti nel dataset: {len(frequenze)}")

    mappa = costruisci_mappatura(frequenze.index)
    mappa["occorrenze"] = mappa["exercise_name"].map(frequenze)
    mappa.to_csv(PROCESSED / "mappatura_esercizi.csv", index=False)

    accettati = mappa["accettato"]
    cop_nomi = accettati.mean()
    cop_righe = mappa.loc[accettati, "occorrenze"].sum() / mappa["occorrenze"].sum()
    print(f"\nSoglia di accettazione: {SOGLIA_ACCETTAZIONE}")
    print(f"Copertura sui nomi distinti ....... {cop_nomi*100:.1f}%")
    print(f"Copertura sulle righe del dataset . {cop_righe*100:.1f}%   <- quella che conta")

    print("\nEsempi di abbinamenti accettati:")
    print(mappa[accettati].nlargest(8, "occorrenze")[
        ["exercise_name", "nome_db", "muscolo_primario", "punteggio", "occorrenze"]].to_string(index=False))

    print("\nEsercizi frequenti NON abbinati (da controllare o mappare a mano):")
    print(mappa[~accettati].nlargest(12, "occorrenze")[
        ["exercise_name", "nome_db", "punteggio", "occorrenze"]].to_string(index=False))

    distrib = (mappa[accettati].groupby("muscolo_primario")["occorrenze"].sum()
               .sort_values(ascending=False).to_frame("occorrenze"))
    distrib["quota_%"] = (distrib["occorrenze"] / distrib["occorrenze"].sum() * 100).round(1)
    print("\nDistribuzione del lavoro per muscolo primario (tutto il catalogo):")
    print(distrib.to_string())

    # --- precisione, non solo copertura ---
    # La copertura non dice se l'abbinamento e' giusto: serve un riferimento annotato a
    # mano. Conta l'insieme B_test, estratto dopo aver scritto le regole. I casi
    # ambigui restano fuori dal denominatore.
    percorso_ann = ROOT / "data" / "annotazione_muscoli.csv"
    if percorso_ann.exists():
        ann = pd.read_csv(percorso_ann).merge(
            mappa[["exercise_name", "muscolo_primario", "accettato", "occorrenze"]],
            on="exercise_name", how="left")
        ann["macro_ottenuto"] = ann["muscolo_primario"].map(MACRO_GRUPPI)
        ann.loc[~ann["accettato"].fillna(False), "macro_ottenuto"] = "escluso"
        ann["giusto"] = ann["macro_ottenuto"] == ann["macro_atteso"]
        valide = ann[ann["macro_atteso"] != "ambiguo"]
        print("\n=== PRECISIONE DELL'ABBINAMENTO (annotazione manuale) ===")
        prec = []
        for nome, g in valide.groupby("campione", sort=False):
            amb = int((ann["campione"] == nome).sum() - len(g))
            riga = {"campione": nome, "nomi_valutati": len(g), "corretti": int(g["giusto"].sum()),
                    "precisione_nomi_%": round(g["giusto"].mean() * 100, 1),
                    "precisione_righe_%": round(g.loc[g["giusto"], "occorrenze"].sum()
                                                / g["occorrenze"].sum() * 100, 1),
                    "ambigui_esclusi": amb}
            lo, hi = intervallo_wilson(riga["corretti"], riga["nomi_valutati"])
            riga["ic_95%"] = f"[{lo*100:.1f}%, {hi*100:.1f}%]"
            prec.append(riga)
        prec = pd.DataFrame(prec)
        print(prec.to_string(index=False))
        print("  L'intervallo di confidenza conta: 40 nomi sono pochi, e la precisione"
              "\n  dichiarata e' il limite inferiore dell'intervallo, non il valore puntuale.")
        sbagliati = valide[~valide["giusto"]]
        if len(sbagliati):
            print("\nAbbinamenti sbagliati nell'annotazione (dichiarati, non corretti a posteriori):")
            print(sbagliati[["campione", "exercise_name", "macro_atteso",
                             "macro_ottenuto", "occorrenze"]].to_string(index=False))
        prec.to_csv(OUT_TAB / "precisione_muscoli.csv", index=False)
    else:
        print(f"\nAVVISO: manca {percorso_ann.relative_to(ROOT)}, quindi la precisione "
              "dell'abbinamento\n  non viene misurata e outputs/tables/precisione_muscoli.csv "
              "non viene scritto.")

    riepilogo = pd.DataFrame([{
        "soglia": SOGLIA_ACCETTAZIONE,
        "nomi_distinti": len(mappa),
        "nomi_abbinati": int(accettati.sum()),
        "copertura_nomi_%": round(cop_nomi * 100, 1),
        "copertura_righe_%": round(cop_righe * 100, 1),
    }])
    riepilogo.to_csv(OUT_TAB / "copertura_muscoli.csv", index=False)
    distrib.to_csv(OUT_TAB / "distribuzione_muscoli.csv")
    print("\nSalvati: data/processed/mappatura_esercizi.csv, outputs/tables/copertura_muscoli.csv")


if __name__ == "__main__":
    main()

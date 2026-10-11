"""02 - Pulizia del dataset e costruzione del catalogo.

Applica le regole di src/catalog.py e salva data/processed/catalogo_pulito.csv e
outputs/tables/report_pulizia.csv.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.catalog import costruisci_catalogo
from src.data_loading import OUT_TAB

PROCESSED = ROOT / "data" / "processed"


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    OUT_TAB.mkdir(parents=True, exist_ok=True)

    catalogo, report = costruisci_catalogo()

    print("=" * 70)
    print("VERSIONI: un programma e' una coppia (titolo, data di creazione)")
    print("=" * 70)
    voci_versioni = ["titoli con piu' versioni", "programmi in piu' grazie alla separazione"]
    for voce in voci_versioni:
        print(f"  {report[voce]:>6}  {voce}")

    print("\n" + "=" * 70)
    print("CORREZIONI A LIVELLO DI RIGA")
    print("=" * 70)
    for motivo, n in report["righe corrette nel dettaglio"].items():
        print(f"  {n:>6}  {motivo}")

    print("\n" + "=" * 70)
    print("PROGRAMMI ESCLUSI (un programma puo' violare piu' regole)")
    print("=" * 70)
    righe = []
    for motivo, n in report["programmi esclusi per motivo"].items():
        print(f"  {n:>6}  {motivo}")
        righe.append({"motivo": motivo, "programmi": n})

    print("\n" + "=" * 70)
    print("RISULTATO")
    print("=" * 70)
    print(f"  programmi di partenza .................. {report['programmi totali']}")
    print(f"  esclusi (conteggio unico) .............. {report['programmi esclusi (unici)']}")
    print(f"  catalogo pulito ........................ {report['programmi nel catalogo pulito']}")
    print(f"  di cui con meno di 3 settimane ......... {report['di cui con progressione non misurabile']}"
          f"  (tenuti, ma senza progressione misurabile)")

    # Salvo il report intero: un programma puo' violare piu' regole, quindi la sola
    # tabella dei motivi non darebbe il numero degli esclusi.
    completo = [{"categoria": "versioni", "voce": voce, "valore": report[voce]}
                for voce in voci_versioni]
    completo += [{"categoria": "correzione su riga", "voce": motivo, "valore": n}
                 for motivo, n in report["righe corrette nel dettaglio"].items()]
    completo += [{"categoria": "esclusione programma", "voce": r["motivo"],
                  "valore": r["programmi"]} for r in righe]
    completo += [
        {"categoria": "totale", "voce": "programmi di partenza",
         "valore": report["programmi totali"]},
        {"categoria": "totale", "voce": "programmi esclusi (conteggio unico)",
         "valore": report["programmi esclusi (unici)"]},
        {"categoria": "totale", "voce": "programmi nel catalogo pulito",
         "valore": report["programmi nel catalogo pulito"]},
        {"categoria": "totale", "voce": "di cui con progressione non misurabile",
         "valore": report["di cui con progressione non misurabile"]},
    ]
    pd.DataFrame(completo).to_csv(OUT_TAB / "report_pulizia.csv", index=False)
    colonne_da_salvare = [c for c in catalogo.columns if c not in ("level_list", "goal_list", "livelli_num")]
    catalogo[colonne_da_salvare].to_csv(PROCESSED / "catalogo_pulito.csv", index=False)
    print("\nSalvati: data/processed/catalogo_pulito.csv e outputs/tables/report_pulizia.csv")


if __name__ == "__main__":
    main()

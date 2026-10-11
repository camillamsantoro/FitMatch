"""14 - Estrae le schede per la demo.

Il dettaglio completo e' troppo grande: tengo solo cio' che serve alla scheda e lo
salvo compresso. Va rieseguito quando cambia il catalogo. gzip con mtime=0 da' lo
stesso file a ogni esecuzione.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loading import load_detail
from src.catalog import pulisci_dettaglio

PROCESSED = ROOT / "data" / "processed"
DESTINAZIONE = PROCESSED / "schede.csv.gz"


def main():
    catalogo = pd.read_csv(PROCESSED / "catalogo_con_qualita.csv")
    # Lo stesso dettaglio pulito degli indicatori: la scheda non puo' mostrare serie
    # diverse da quelle contate dal sistema.
    det, _ = pulisci_dettaglio(load_detail())
    det = det[det["title"].isin(set(catalogo["title"]))]
    det = det[["title", "week", "day", "exercise_name", "sets",
               "reps_count", "duration_s", "intensity"]]
    for colonna in ["week", "day", "sets", "reps_count", "duration_s", "intensity"]:
        det[colonna] = pd.to_numeric(det[colonna], downcast="float")

    print(f"Righe: {len(det):,}".replace(",", ".") + f" | programmi: {det['title'].nunique()}")
    det.to_csv(DESTINAZIONE, index=False, compression={"method": "gzip", "mtime": 0})
    print(f"Salvato {DESTINAZIONE.relative_to(ROOT)} "
          f"({DESTINAZIONE.stat().st_size / 1e6:.0f} MB)")


if __name__ == "__main__":
    main()

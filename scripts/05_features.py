"""05 - Indicatori di qualita' dei programmi.

Calcola i sei indicatori, li unisce al catalogo e salva
data/processed/catalogo_con_qualita.csv con alcune tabelle di controllo.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loading import load_detail, OUT_TAB
from src.grafici import (applica_stile, numero, virgola, pulisci, salva,
                         NOMI_INDICATORI, BLU)
from src.catalog import pulisci_dettaglio
from src.quality import calcola_indicatori, quote_muscolari, PESI_QUALITA

PROCESSED = ROOT / "data" / "processed"


def main():
    catalogo = pd.read_csv(PROCESSED / "catalogo_pulito.csv")
    mappa = pd.read_csv(PROCESSED / "mappatura_esercizi.csv")
    # Dettaglio gia' pulito: le correzioni di src/catalog.py devono valere anche qui.
    dettaglio, _ = pulisci_dettaglio(load_detail())
    dettaglio = dettaglio[dettaglio["title"].isin(set(catalogo["title"]))]

    print(f"Programmi: {len(catalogo)} | righe di esercizi: {len(dettaglio)}")
    ind = calcola_indicatori(dettaglio, mappa, catalogo)

    # Quote di lavoro per gruppo, usate dall'enfasi e dal simulatore. Il calcolo sta
    # solo in src/quality.py.
    quote = quote_muscolari(dettaglio, mappa)
    # Qui i programmi senza muscoli abbinati vanno a zero, ma nel catalogo le loro
    # quote restano mancanti per via del merge "left".
    quote = quote.fillna(0)
    quote.to_csv(PROCESSED / "quote_muscolari.csv")
    print(f"Quote muscolari salvate per {len(quote)} programmi "
          f"su {len(catalogo)} del catalogo "
          f"({len(catalogo) - len(quote)} senza nessun esercizio abbinato a un muscolo)")
    ind = ind.merge(quote.add_prefix("quota_"), left_on="title", right_index=True, how="left")

    componenti = ["q_volume", "q_frequenza", "q_progressione",
                  "q_equilibrio", "q_varieta", "q_coerenza"]
    print("\n=== DISTRIBUZIONE DEGLI INDICATORI (0 = scarso, 1 = ottimo) ===")
    print(ind[componenti + ["qualita"]].describe().round(3).to_string())
    print("\nValori non calcolabili per componente:")
    print(ind[componenti].isna().sum().to_string())

    print("\n=== VALORI GREZZI ===")
    grezzi = ["volume_settimanale", "frequenza_media", "pendenza_rpe",
              "distanza_equilibrio", "varieta"]
    print(ind[grezzi].describe().round(3).to_string())

    print("\n=== CORRELAZIONE TRA COMPONENTI ===")
    print("(se fossero tutte molto correlate, misurerebbero la stessa cosa)")
    print(ind[componenti].corr().round(2).to_string())

    # pendenza_rpe e varieta' quality.py le copia dal catalogo: tolgo il doppione.
    duplicate = [c for c in ind.columns if c in catalogo.columns and c != "title"]
    cat = catalogo.drop(columns=duplicate).merge(ind, on="title", how="left")
    cat.to_csv(PROCESSED / "catalogo_con_qualita.csv", index=False)

    colonne_vista = ["title", "qualita", "q_volume", "q_frequenza", "q_progressione",
                     "q_equilibrio", "q_varieta", "q_coerenza", "settimane",
                     "giorni_settimana", "time_per_workout"]
    print("\n=== 5 PROGRAMMI CON QUALITA' PIU' ALTA ===")
    print(cat.nlargest(5, "qualita")[colonne_vista].round(2).to_string(index=False))
    print("\n=== 5 PROGRAMMI CON QUALITA' PIU' BASSA ===")
    print(cat.nsmallest(5, "qualita")[colonne_vista].round(2).to_string(index=False))

    # --- le premesse della domanda di ricerca ---
    # Se la qualita' dipendesse dai metadati non aggiungerebbe niente. Salvo i numeri
    # in premesse_qualita.csv perche' si possano controllare.
    print("\n=== LA QUALITA' E' INDIPENDENTE DAI METADATI? ===")
    print("(se lo e', aggiunge informazione che il sistema base non ha)")
    premesse = []
    # Oltre ai quattro metadati del modulo guardo settimane e numero di obiettivi,
    # perche' varieta' e coerenza ci sono legate.
    colonne_goal = [c for c in cat.columns if c.startswith("goal_")]
    cat["n_obiettivi"] = cat[colonne_goal].sum(axis=1)
    for col in ["time_per_workout", "program_length", "livello_min", "total_exercises",
                "settimane", "n_obiettivi"]:
        r = float(cat["qualita"].corr(cat[col]))
        premesse.append({"misura": f"correlazione qualita ~ {col}", "valore": round(r, 3)})
        print(f"  correlazione qualita' ~ {col}: {r:.3f}")

    # dice se le sei componenti sono ridondanti
    corr = ind[componenti].corr()
    senza_diagonale = corr.abs().where(~np.eye(len(componenti), dtype=bool))
    coppia = senza_diagonale.stack().idxmax()
    premesse.append({"misura": "correlazione massima fra due componenti",
                     "valore": round(float(senza_diagonale.stack().max()), 3)})
    premesse.append({"misura": f"coppia piu' correlata ({coppia[0]} e {coppia[1]})",
                     "valore": round(float(corr.loc[coppia]), 3)})
    print(f"\n  correlazione massima fra componenti: "
          f"{senza_diagonale.stack().max():.3f} ({coppia[0]} e {coppia[1]})")

    # i comportamenti descritti in quality.py, salvati in tabella
    mis = ind[ind["q_progressione"].notna()]
    fre = ind[ind["q_frequenza"].notna()]
    premesse += [
        {"misura": "programmi con progressione misurabile", "valore": len(mis)},
        {"misura": "programmi penalizzati per progressione", "valore": int((mis["q_progressione"] < 0.25).sum())},
        {"misura": "programmi con frequenza misurabile", "valore": len(fre)},
        {"misura": "programmi con frequenza al voto pieno", "valore": int((fre["q_frequenza"] >= 1).sum())},
        {"misura": "correlazione coerenza ~ numero di obiettivi",
         "valore": round(float(pd.Series(
             cat.set_index("title")["n_obiettivi"].reindex(ind["title"]).values,
             index=ind.index).corr(ind["q_coerenza"])), 3)},
        {"misura": "programmi con piu' di un obiettivo dichiarato", "valore": int((cat["n_obiettivi"] > 1).sum())},
    ]
    pd.DataFrame(premesse).to_csv(OUT_TAB / "premesse_qualita.csv", index=False)
    print("  Salvato: outputs/tables/premesse_qualita.csv")

    # col titolo, per sapere di quale programma e' ogni riga
    ind[["title"] + componenti + ["qualita"]].to_csv(OUT_TAB / "indicatori_qualita.csv", index=False)
    ind[componenti + ["qualita"]].describe().to_csv(OUT_TAB / "statistiche_qualita.csv")
    corr.to_csv(OUT_TAB / "correlazioni_qualita.csv")

    # Sette istogrammi in una griglia 2x4: l'ottavo riquadro va nascosto. Nel titolo
    # quanti programmi hanno il voto, perche' non tutti sono misurabili.
    applica_stile()
    da_disegnare = componenti + ["qualita"]
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.4))
    for ax, col in zip(axes.flat, da_disegnare):
        valori = ind[col].dropna()
        ax.hist(valori, bins=25, range=(0, 1), color=BLU, edgecolor="white", linewidth=0.4)
        ax.set_title(f"{NOMI_INDICATORI[col]} ({numero(len(valori), 0, migliaia=True)} programmi)",
                     loc="left", fontsize=10.5)
        ax.set_xlabel("voto (0 = scarso, 1 = ottimo)")
        ax.set_ylabel("programmi")
        virgola(ax, "both")
        pulisci(ax, griglia="y")
    for ax in axes.flat[len(da_disegnare):]:
        ax.set_visible(False)
    fig.suptitle("Distribuzione dei sei indicatori di qualità e del voto complessivo",
                 fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    salva(fig, "distribuzione_qualita")

    print(f"\nPesi usati: {PESI_QUALITA}")
    print("Salvati: data/processed/catalogo_con_qualita.csv, outputs/tables/indicatori_qualita.csv")


if __name__ == "__main__":
    main()

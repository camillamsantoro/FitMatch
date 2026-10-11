"""17 - Le quattro figure principali del progetto.

confronto_sistemi.png, ablation.png, robustezza.png e guadagno_per_utente.png. Le
altre figure le fanno gli script delle singole analisi; lo stile comune e' in
src/grafici.py. Barre orizzontali perche' i nomi sono lunghi, intervalli sempre
visibili, e categorie distinte anche dalla forma per la stampa in bianco e nero.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loading import OUT_TAB
from src.grafici import (applica_stile, numero, virgola, pulisci, intervallo, errori,
                         salva, NOMI_SISTEMI, NOMI_SCENARI, NOMI_INDICATORI, BLU,
                         BLU_CHIARO, ROSSO, VERDE, GRIGIO, INCHIOSTRO, INCHIOSTRO_2)

PROPOSTI = {"rilevanza graduale", "completo (rilevanza + qualita)"}

applica_stile()


# --- figura 1 ---
def figura_sistemi():
    d = pd.read_csv(OUT_TAB / "valutazione_sistemi.csv").sort_values("ndcg")
    ic = [intervallo(t) for t in d["ndcg_ic"]]
    colori = [BLU if s in PROPOSTI else BLU_CHIARO for s in d["sistema"]]
    caso = float(d.loc[d["sistema"] == "casuale", "ndcg"].iloc[0])

    fig, ax = plt.subplots(figsize=(9, 4.6))
    y = range(len(d))
    ax.barh(y, d["ndcg"], xerr=errori(list(d["ndcg"]), ic), color=colori,
            height=0.62, error_kw={"ecolor": INCHIOSTRO_2, "elinewidth": 1.4,
                                   "capsize": 3, "capthick": 1.4})
    ax.axvline(caso, color=INCHIOSTRO_2, linewidth=1, linestyle=(0, (4, 3)), zorder=0)
    ax.text(caso, -0.72, "  livello del caso", fontsize=9,
            color=INCHIOSTRO_2, va="center")

    for i, (v, (lo, hi)) in enumerate(zip(d["ndcg"], ic)):
        x = hi + 0.012
        # un'etichetta che cadrebbe sulla linea del caso va oltre
        if x - 0.006 < caso < x + 0.045:
            x = caso + 0.010
        ax.text(x, i, numero(v, 3), va="center", fontsize=9.5, color=INCHIOSTRO,
                fontweight="bold" if d["sistema"].iloc[i] in PROPOSTI else "normal")

    ax.set_yticks(list(y))
    ax.set_yticklabels([NOMI_SISTEMI.get(s, s) for s in d["sistema"]])
    ax.set_xlim(0, 0.93)
    ax.set_ylim(-1.0, len(d) - 0.35)
    virgola(ax, "x", 1)
    ax.set_xlabel("NDCG@10 — media su 300 utenti simulati, barre = intervallo di confidenza 95%")
    ax.set_title("Quanto valgono i sistemi a confronto", fontsize=13,
                 fontweight="bold", loc="left", pad=14)
    ax.legend(handles=[Patch(color=BLU, label="sistema proposto"),
                       Patch(color=BLU_CHIARO, label="termine di paragone")],
              loc="lower right", frameon=False, fontsize=9)
    pulisci(ax)
    ax.spines["left"].set_visible(False)
    salva(fig, "confronto_sistemi")


# --- figura 2 ---
# Il colore dipende dall'intervallo, non dal segno: se l'intervallo contiene lo zero
# l'indicatore non danneggia, non si distingue da zero.
CATEGORIE_ABLATION = {
    "danneggia": (ROSSO, "///", "toglierlo migliora il sistema (significativo): l'indicatore danneggia"),
    "serve": (VERDE, "xxx", "toglierlo peggiora il sistema (significativo): l'indicatore serve"),
    "neutro": (GRIGIO, "", "non distinguibile da zero"),
}


def figura_ablation():
    d = pd.read_csv(OUT_TAB / "ablation_qualita.csv")
    d = d[d["configurazione"].str.startswith("senza ")].copy()
    # La differenza viene dalla tabella, non dagli NDCG arrotondati.
    d["differenza"] = d["differenza_ndcg"]
    d["nome"] = d["configurazione"].str.replace("senza ", "", regex=False)
    d = d.sort_values("differenza")
    ic = [intervallo(t) for t in d["ic_differenza"]]
    categorie = ["danneggia" if lo > 0 else "serve" if hi < 0 else "neutro" for lo, hi in ic]

    fig, ax = plt.subplots(figsize=(9, 3.9))
    y = list(range(len(d)))
    for i, (v, (lo, hi), cat) in enumerate(zip(d["differenza"], ic, categorie)):
        colore, tratteggio, _ = CATEGORIE_ABLATION[cat]
        ax.barh(i, v, color=colore, height=0.6, hatch=tratteggio,
                edgecolor=INCHIOSTRO_2 if tratteggio else colore, linewidth=0.6)
        ax.errorbar(v, i, xerr=[[v - lo], [hi - v]], fmt="none", ecolor=INCHIOSTRO,
                    elinewidth=1.4, capsize=3, capthick=1.4)
        if v < 0:
            ax.text(min(lo, v) - 0.003, i, numero(v, 4, segno=True), va="center",
                    ha="right", fontsize=9.5, color=INCHIOSTRO)
        else:
            ax.text(max(hi, v) + 0.003, i, numero(v, 4, segno=True), va="center",
                    fontsize=9.5, color=INCHIOSTRO)
    ax.axvline(0, color=INCHIOSTRO, linewidth=1.2)

    ax.set_yticks(y)
    ax.set_yticklabels([NOMI_INDICATORI.get(n, n) for n in d["nome"]])
    ax.set_xlim(-0.085, 0.05)
    virgola(ax, "x", 2)
    ax.set_xlabel("variazione dell'NDCG@10 togliendo l'indicatore — barre = intervallo 95%")
    ax.set_title("Quale indicatore di qualità serve davvero", fontsize=13,
                 fontweight="bold", loc="left", pad=14)
    presenti = [c for c in CATEGORIE_ABLATION if c in categorie]
    fig.legend(handles=[Patch(facecolor=CATEGORIE_ABLATION[c][0], hatch=CATEGORIE_ABLATION[c][1],
                              edgecolor=INCHIOSTRO_2, label=CATEGORIE_ABLATION[c][2])
                        for c in presenti],
               loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=1, frameon=False, fontsize=9)
    pulisci(ax)
    ax.spines["left"].set_visible(False)
    fig.tight_layout()
    salva(fig, "ablation")


# --- figura 3 ---
def figura_robustezza():
    """Disegna il guadagno sulla rilevanza e il contributo sopra il metadato.

    Pieno vuol dire significativo, vuoto no. Il conteggio degli scenari significativi
    non va nel titolo: dipende dall'estrazione del rumore, e la nota lo dice.
    """
    d = pd.read_csv(OUT_TAB / "robustezza.csv")
    m = pd.read_csv(OUT_TAB / "robustezza_metadato.csv")
    semi = pd.read_csv(OUT_TAB / "semi_rumore.csv")
    realizzazioni = semi[semi["realizzazione"] != "minimo / mediana / massimo"]
    n_real = len(realizzazioni)
    min_sig = int(realizzazioni["scenari_misto_significativo"].astype(int).min())
    max_sig = int(realizzazioni["scenari_misto_significativo"].astype(int).max())

    pannelli = [
        (d["guadagno qualita'"], d["ic guadagno"], d["scenario"],
         "sulla sola rilevanza", 3),
        (m["misto_meno_metadato"], m["ic_misto_meno_metadato"], m["scenario"],
         "sopra il numero di esercizi (misto − metadato)", 4),
    ]
    fig, assi = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    conteggi = []
    for ax, (valori, testi_ic, scenari, titolo, cifre) in zip(assi, pannelli):
        valori = list(valori)[::-1]
        ic = [intervallo(t) for t in list(testi_ic)[::-1]]
        conteggi.append(sum(1 for lo, _ in ic if lo > 0))
        ax.axvline(0, color=INCHIOSTRO, linewidth=1.2)
        for i, (v, (lo, hi)) in enumerate(zip(valori, ic)):
            significativo = lo > 0
            colore = BLU if significativo else ROSSO
            ax.errorbar([v], [i], xerr=[[v - lo], [hi - v]], fmt="o", color=colore,
                        markersize=8, elinewidth=1.6, capsize=4, capthick=1.6,
                        markerfacecolor=colore if significativo else "white",
                        markeredgecolor=colore, markeredgewidth=1.6)
            ax.text(hi + 0.0025, i, numero(v, cifre, segno=True), va="center",
                    fontsize=9.5, color=INCHIOSTRO)
        ax.set_yticks(list(range(len(valori))))
        ax.set_yticklabels([NOMI_SCENARI.get(s, s) for s in list(scenari)[::-1]])
        ax.set_xlim(-0.008, 0.072)
        ax.set_ylim(-0.7, len(valori) - 0.3)
        virgola(ax, "x", 2)
        ax.set_title(titolo, fontsize=10.5, loc="left", color=INCHIOSTRO_2)
        ax.set_xlabel("differenza di NDCG@10 — barre = intervallo 95%")
        pulisci(ax)
    fig.suptitle("Quanto regge il guadagno della qualità negli scenari di robustezza",
                 fontsize=13, fontweight="bold", x=0.01, ha="left")
    nota = (f"In questa realizzazione del rumore di gusto il guadagno è significativo in "
            f"{conteggi[0]} scenari su {len(d)} e il contributo sopra il metadato in "
            f"{conteggi[1]} su {len(m)}.\nCon {n_real} realizzazioni diverse il secondo "
            f"conteggio va da {min_sig} a {max_sig} su {len(m)} (semi_rumore.csv): "
            f"è un valore di questa realizzazione, non una proprietà del contributo.")
    fig.legend(handles=[
        Line2D([], [], marker="o", linestyle="none", markersize=8, color=BLU,
               markerfacecolor=BLU, label="significativo: intervallo tutto sopra lo zero"),
        Line2D([], [], marker="o", linestyle="none", markersize=8, color=ROSSO,
               markerfacecolor="white", markeredgewidth=1.6,
               label="non significativo: intervallo che tocca lo zero")],
        loc="upper left", bbox_to_anchor=(0.01, 0.0), ncol=2, frameon=False, fontsize=9)
    fig.text(0.01, -0.07, nota, fontsize=9, color=INCHIOSTRO_2, ha="left", va="top")
    fig.tight_layout()
    salva(fig, "robustezza")


# --- figura 4 ---
def figura_guadagno_per_utente():
    """Disegna le differenze di NDCG@10 fra completo e rilevanza, utente per utente.

    Fa vedere cosa c'e' sotto la media: per una parte degli utenti il guadagno e'
    negativo.
    """
    d = np.load(ROOT / "data" / "processed" / "differenze_qualita.npy")
    sintesi = pd.read_csv(OUT_TAB / "confronto_appaiato.csv").iloc[0]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    bordi = np.linspace(-0.28, 0.22, 51)
    conteggi, _ = np.histogram(d, bins=bordi)
    centri = (bordi[:-1] + bordi[1:]) / 2
    larghezza = (bordi[1] - bordi[0]) * 0.86
    negativi = centri < 0
    ax.bar(centri[negativi], conteggi[negativi], width=larghezza, color=ROSSO,
           hatch="///", edgecolor="white", linewidth=0.5)
    ax.bar(centri[~negativi], conteggi[~negativi], width=larghezza, color=BLU,
           edgecolor="white", linewidth=0.5)
    ax.axvline(0, color=INCHIOSTRO_2, linewidth=1)
    media = float(sintesi["differenza_media"])
    alto = conteggi.max()
    # etichetta della media sopra le barre: dentro il picco coprirebbe due colonne
    ax.set_ylim(0, alto * 1.22)
    ax.axvline(media, color=INCHIOSTRO, linewidth=1, linestyle=(0, (3, 2)))
    ax.text(media + 0.004, alto * 1.17, f"media {numero(media, 3, segno=True)}",
            va="top", ha="left", fontsize=9, color=INCHIOSTRO)
    ax.text(-0.27, alto * 0.62,
            f"peggiorano\n{numero(sintesi['utenti_peggiorati_%'], 1)}%",
            ha="left", va="center", fontsize=10, color=INCHIOSTRO_2)
    ax.text(0.21, alto * 0.62,
            f"migliorano\n{numero(sintesi['utenti_migliorati_%'], 1)}%",
            ha="right", va="center", fontsize=10, color=INCHIOSTRO_2)
    virgola(ax, "x", 1)
    ax.set_xlabel("NDCG@10 del sistema completo meno NDCG@10 della sola rilevanza, per utente")
    ax.set_ylabel("utenti")
    ax.set_title("Il guadagno della qualità, utente per utente (300 utenti simulati)",
                 loc="left", fontsize=11)
    pulisci(ax, griglia="y")
    salva(fig, "guadagno_per_utente")


if __name__ == "__main__":
    # robustezza.png ha bisogno di semi_rumore.csv di 22: meglio fermarsi prima di
    # scrivere le figure che a meta'.
    if not (OUT_TAB / "semi_rumore.csv").exists():
        sys.exit("17_figure.py: manca outputs/tables/semi_rumore.csv, che serve alla nota di "
                 "robustezza.png. Eseguire prima scripts/22_semi.py (ordine del README).")
    print("Figure generate in outputs/figures/:")
    figura_sistemi()
    figura_ablation()
    figura_robustezza()
    figura_guadagno_per_utente()

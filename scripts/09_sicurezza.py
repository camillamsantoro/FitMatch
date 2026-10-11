"""09 - Il secondo asse: gradimento contro adeguatezza del programma.

Per ogni lambda guarda la top-10: qualita' media, quota di programmi in sovraccarico
e probabilita' di abbandono. La qualita' sale con lambda per costruzione, quindi non
prova niente. Il sovraccarico mostra il prezzo che l'NDCG nasconde.
Salva tre tabelle in outputs/tables e gradimento_vs_sovraccarico.png.
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
from src.data_loading import OUT_TAB
from src.grafici import (applica_stile, numero, virgola, pulisci, salva, intervallo,
                         errori, BLU, ROSSO, INCHIOSTRO_2)
from src.simulator import genera_utenti, probabilita_rifiuto, SOGLIA_SOVRACCARICO, carica_verita
from src.recommenders import NON_FATTIBILE
from src.scoring import punteggio
from src.evaluation import ndcg_at_k, bootstrap_ic, ordina

PROCESSED = ROOT / "data" / "processed"
K = 10
LAMBDA_PROVATI = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
LAMBDA_USATO = 0.3
SEME_TARATURA = 7  # lo stesso di 21
# La soglia 1,2 e' una scelta: con altre soglie cambiano i livelli, non la tendenza.
SOGLIE_PROVATE = [1.1, 1.2, 1.3, 1.5]


def misura(catalogo, utenti, lam, U=None):
    """Calcola per ogni utente NDCG, qualita', carico e abbandono della top-10."""
    ndcg, qualita, rapp_vero, rapp_stima, rifiuto = [], [], [], [], []
    for i, utente in enumerate(utenti):
        profilo = utente.profilo_dichiarato()
        p = punteggio(catalogo, profilo, lambda_qualita=lam).fillna(NON_FATTIBILE).values
        top = ordina(p, K)
        if U is not None:
            ndcg.append(ndcg_at_k(p, U[i], K))
        qualita.append(float(catalogo["qualita"].iloc[top].mean()))
        volume = catalogo["volume_settimanale"].iloc[top].values
        rapp_vero.append(volume / utente.capacita_serie)
        rapp_stima.append(volume / max(profilo.capacita_stimata, 1e-6))
        rifiuto.append(float(probabilita_rifiuto(catalogo, utente).iloc[top].mean()))
    return {
        "ndcg": np.array(ndcg, dtype=float),
        "qualita": np.array(qualita),
        "rapporto_vero": np.vstack(rapp_vero),
        "rapporto_stima": np.vstack(rapp_stima),
        "abbandono": np.array(rifiuto),
    }


def quota(rapporti, soglia=SOGLIA_SOVRACCARICO):
    """Calcola per ogni utente la quota (in %) di consigliati oltre la soglia."""
    return (rapporti > soglia).mean(axis=1) * 100


def ic(valori, cifre):
    _, lo, hi = bootstrap_ic(valori)
    return f"[{lo:.{cifre}f}, {hi:.{cifre}f}]"


def main():
    catalogo = pd.read_csv(PROCESSED / "catalogo_con_qualita.csv")
    U, utenti = carica_verita(catalogo, PROCESSED)
    utenti7 = genera_utenti(300, seme=SEME_TARATURA)

    misure = {lam: misura(catalogo, utenti, lam, U) for lam in LAMBDA_PROVATI}
    misure7 = {lam: misura(catalogo, utenti7, lam) for lam in LAMBDA_PROVATI}

    righe = []
    for lam, m in misure.items():
        m_ndcg, b_ndcg, a_ndcg = bootstrap_ic(m["ndcg"])
        sovr = quota(m["rapporto_vero"])
        stima = quota(m["rapporto_stima"])
        righe.append({
            "lambda": lam,
            "ndcg": round(m_ndcg, 3), "ndcg_ic": f"[{b_ndcg:.3f}, {a_ndcg:.3f}]",
            "qualita_media_top10": round(float(np.mean(m["qualita"])), 3),
            "quota_sovraccarico_%": round(float(np.mean(sovr)), 1),
            "probabilita_abbandono": round(float(np.mean(m["abbandono"])), 3),
            "qualita_ic": ic(m["qualita"], 3),
            "sovraccarico_ic": ic(sovr, 1),
            "abbandono_ic": ic(m["abbandono"], 3),
            "quota_oltre_stima_%": round(float(np.mean(stima)), 1),
        })
        r = righe[-1]
        print(f"lambda {lam:.1f} | NDCG@10 {m_ndcg:.3f} | qualita' top-10 "
              f"{r['qualita_media_top10']:.3f} | sovraccarico "
              f"{r['quota_sovraccarico_%']:.1f}% {r['sovraccarico_ic']} | abbandono "
              f"{r['probabilita_abbandono']:.3f} | oltre la stima del sistema "
              f"{r['quota_oltre_stima_%']:.1f}%")
    tab = pd.DataFrame(righe)
    tab.to_csv(OUT_TAB / "gradimento_vs_adeguatezza.csv", index=False)

    # --- confronti appaiati con lambda = 0,3, su due popolazioni ---
    print(f"\nConfronti appaiati con lambda = {LAMBDA_USATO} "
          f"(differenza = altro lambda - {LAMBDA_USATO}):")
    appaiati = []
    for seme, mis in [(42, misure), (SEME_TARATURA, misure7)]:
        rif = mis[LAMBDA_USATO]
        for lam in LAMBDA_PROVATI:
            if lam == LAMBDA_USATO:
                continue
            d_sovr = quota(mis[lam]["rapporto_vero"]) - quota(rif["rapporto_vero"])
            d_abb = mis[lam]["abbandono"] - rif["abbandono"]
            ms, ls, hs = bootstrap_ic(d_sovr)
            ma, la, ha = bootstrap_ic(d_abb)
            appaiati.append({
                "seme": seme, "lambda": lam,
                "diff_sovraccarico_punti": round(ms, 2),
                "ic_sovraccarico": f"[{ls:+.2f}, {hs:+.2f}]",
                "diff_abbandono": round(ma, 3),
                "ic_abbandono": f"[{la:+.3f}, {ha:+.3f}]",
            })
            if lam in (0.0, 0.2, 0.4, 0.5):
                print(f"  seme {seme:>2} lambda {lam:.1f}: sovraccarico {ms:+.2f} punti "
                      f"[{ls:+.2f}, {hs:+.2f}] | abbandono {ma:+.3f} [{la:+.3f}, {ha:+.3f}]")
    pd.DataFrame(appaiati).to_csv(OUT_TAB / "secondo_asse_appaiato.csv", index=False)

    # --- sensibilita' alla soglia ---
    righe_soglia = []
    for soglia in SOGLIE_PROVATE:
        riga = {"soglia": soglia}
        for lam in LAMBDA_PROVATI:
            riga[f"lambda_{lam}"] = round(float(np.mean(quota(misure[lam]["rapporto_vero"],
                                                              soglia))), 1)
        righe_soglia.append(riga)
    soglie = pd.DataFrame(righe_soglia)
    soglie.to_csv(OUT_TAB / "soglia_sovraccarico.csv", index=False)
    crescente = all(np.all(np.diff(r[1:].values) >= 0) for _, r in soglie.iterrows())
    print("\nSovraccarico (%) al variare della soglia, per lambda:")
    print(soglie.to_string(index=False))
    print(f"  cresce con lambda per ogni soglia: {crescente}")

    figura_secondo_asse(tab)
    print("\nSalvati: outputs/tables/gradimento_vs_adeguatezza.csv, secondo_asse_appaiato.csv, "
          "soglia_sovraccarico.csv, outputs/figures/gradimento_vs_sovraccarico.png")


def figura_secondo_asse(tab):
    """Disegna un punto per lambda: sovraccarico in ascissa, gradimento in ordinata.

    In ordinata non metto la qualita', che sale con lambda per costruzione e non
    mostrerebbe nessun prezzo.
    """
    applica_stile()
    x = list(tab["quota_sovraccarico_%"])
    y = list(tab["ndcg"])
    ic_x = [intervallo(s) for s in tab["sovraccarico_ic"]]
    ic_y = [intervallo(s) for s in tab["ndcg_ic"]]
    lam = list(tab["lambda"])

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(x, y, color=BLU, linewidth=1.2, zorder=1)
    ax.errorbar(x, y, xerr=errori(x, ic_x), yerr=errori(y, ic_y), fmt="o", color=BLU,
                markersize=6, elinewidth=1.1, capsize=2.5, zorder=2)
    i3 = lam.index(LAMBDA_USATO)
    ax.plot([x[i3]], [y[i3]], marker="o", markersize=13, markerfacecolor="none",
            markeredgecolor=ROSSO, markeredgewidth=1.8, zorder=3)
    # I primi punti sono vicini in alto: etichette sotto, su due righe, col filo.
    posizioni = {0.0: (-24, -45), 0.1: (14, -95), 0.2: (12, -55), 0.3: (18, -115),
                 0.4: (22, -55), 0.5: (24, -75), 0.7: (14, 22), 1.0: (-70, 16)}
    for l, xi, yi in zip(lam, x, y):
        testo = f"λ = {numero(l, 1)}" + (" (usato)" if l == LAMBDA_USATO else "")
        usato = l == LAMBDA_USATO
        ax.annotate(testo, (xi, yi), xytext=posizioni.get(l, (10, -40)),
                    textcoords="offset points", fontsize=9, ha="center", va="center",
                    color=ROSSO if usato else INCHIOSTRO_2,
                    fontweight="bold" if usato else "normal",
                    arrowprops={"arrowstyle": "-", "color": ROSSO if usato else INCHIOSTRO_2,
                                "linewidth": 0.7, "shrinkA": 2, "shrinkB": 6})
    ax.set_xlabel(f"consigli oltre {numero(SOGLIA_SOVRACCARICO, 1)} volte la capacità vera "
                  "della persona (%) — barre = intervallo 95%")
    ax.set_ylabel("gradimento simulato (NDCG@10) — barre = intervallo 95%")
    ax.set_title("Il secondo asse: quanto gradimento si compra con quanto sovraccarico",
                 loc="left", fontsize=12, fontweight="bold")
    virgola(ax, "x", 0)
    virgola(ax, "y", 2)
    pulisci(ax, griglia="both")
    q0, q1 = tab["qualita_media_top10"].iloc[0], tab["qualita_media_top10"].iloc[-1]
    fig.text(0.01, -0.01, "La qualità media della top-10 (da " + numero(q0, 3) + " a λ = 0 a "
             + numero(q1, 3) + " a λ = 1) sale con λ per costruzione: è la grandezza che "
             "λ pesa, e per questo non è un asse della figura.",
             fontsize=8.5, color=INCHIOSTRO_2, ha="left", va="top")
    fig.tight_layout()
    salva(fig, "gradimento_vs_sovraccarico")


if __name__ == "__main__":
    main()

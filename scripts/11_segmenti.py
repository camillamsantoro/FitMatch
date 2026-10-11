"""11 - Analisi per segmenti di utenti.

La media puo' nascondere un gruppo servito male, come chi si allena a casa. Le
tabelle segmenti_{attrezzatura,livello,obiettivo}.csv sono solo medie; i test stanno
in segmenti_guadagno.csv (appaiati, con Bonferroni) e segmenti_confronti.csv (a due
campioni, perche' gli utenti sono diversi).
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
from src.grafici import (applica_stile, numero, virgola, pulisci, salva,
                         NOMI_ATTREZZATURE, BLU, BLU_CHIARO, VERDE, ROSSO,
                         INCHIOSTRO_2)
from src.simulator import probabilita_rifiuto, carica_verita, SOGLIA_SOVRACCARICO
from src.recommenders import NON_FATTIBILE
from src.scoring import punteggio, rilevanza
from src.evaluation import (ndcg_at_k, precision_at_k, bootstrap_ic, ordina,
                            bootstrap_ic_due_campioni)

PROCESSED = ROOT / "data" / "processed"
K = 10
MIN_UTENTI_IC = 20
DIMENSIONI = ["attrezzatura", "livello", "obiettivo"]
CONFRONTI = [
    ("NDCG, solo manubri - palestra", "ndcg_completo", "attrezzatura", "Dumbbell Only", "Full Gym"),
    ("qualita' top-10, solo manubri - palestra", "qualita_top10", "attrezzatura",
     "Dumbbell Only", "Full Gym"),
    ("sovraccarico, solo manubri - palestra", "sovraccarico", "attrezzatura",
     "Dumbbell Only", "Full Gym"),
    ("guadagno, palestra - garage", "guadagno_qualita", "attrezzatura", "Full Gym", "Garage Gym"),
    ("sovraccarico, principianti - avanzati", "sovraccarico", "livello", "Beginner", "Advanced"),
    ("guadagno, novizi - principianti", "guadagno_qualita", "livello", "Novice", "Beginner"),
]


def main():
    catalogo = pd.read_csv(PROCESSED / "catalogo_con_qualita.csv")
    U, utenti = carica_verita(catalogo, PROCESSED)

    righe = []
    for i, utente in enumerate(utenti):
        profilo = utente.profilo_dichiarato()
        p_tot = punteggio(catalogo, profilo).fillna(NON_FATTIBILE).values
        p_ril = rilevanza(catalogo, profilo).fillna(NON_FATTIBILE).values
        top = ordina(p_tot, K)
        rapporto = catalogo["volume_settimanale"].values[top] / utente.capacita_serie
        righe.append({
            "utente": i,
            "attrezzatura": utente.attrezzatura,
            "livello": utente.livello,
            "obiettivo": utente.obiettivo,
            "giorni": utente.giorni_settimana,
            "minuti": utente.minuti_per_seduta,
            "ndcg_completo": ndcg_at_k(p_tot, U[i], K),
            "ndcg_rilevanza": ndcg_at_k(p_ril, U[i], K),
            "precision_completo": precision_at_k(p_tot, U[i], K),
            "qualita_top10": float(catalogo["qualita"].iloc[top].mean()),
            "sovraccarico": float((rapporto > SOGLIA_SOVRACCARICO).mean()),
            "abbandono": float(probabilita_rifiuto(catalogo, utente).iloc[top].mean()),
            "programmi_fattibili": int((U[i] > 0).sum()),
        })
    df = pd.DataFrame(righe)
    df["guadagno_qualita"] = df["ndcg_completo"] - df["ndcg_rilevanza"]
    # serve la sola rilevanza per separare il sovraccarico del catalogo da quello
    # aggiunto dalla qualita'.
    sovr_ril = np.array([
        float((catalogo["volume_settimanale"].values[ordina(
            rilevanza(catalogo, u.profilo_dichiarato()).fillna(NON_FATTIBILE).values, K)]
            / u.capacita_serie > SOGLIA_SOVRACCARICO).mean())
        for u in utenti])

    for colonna in ["attrezzatura", "livello", "obiettivo"]:
        print("\n" + "=" * 96)
        print(f"PER {colonna.upper()}")
        print("=" * 96)
        tab = df.groupby(colonna).agg(
            utenti=("utente", "count"),
            programmi_fattibili=("programmi_fattibili", "median"),
            ndcg_completo=("ndcg_completo", "mean"),
            ndcg_rilevanza=("ndcg_rilevanza", "mean"),
            guadagno=("guadagno_qualita", "mean"),
            precision=("precision_completo", "mean"),
            qualita_top10=("qualita_top10", "mean"),
            sovraccarico=("sovraccarico", "mean"),
            abbandono=("abbandono", "mean"),
        ).round(3).sort_values("ndcg_completo")
        print(tab.to_string())
        tab.to_csv(OUT_TAB / f"segmenti_{colonna}.csv")

    print("\n" + "=" * 96)
    print("IL GUADAGNO DELLA QUALITA' E' SIGNIFICATIVO IN OGNI SEGMENTO?")
    print("(95% e Bonferroni per il numero di segmenti della dimensione)")
    print("=" * 96)
    df_test = df.assign(diff_sovraccarico=(df["sovraccarico"] - sovr_ril) * 100)
    righe_test = []
    for dimensione in DIMENSIONI:
        gruppi = list(df_test.groupby(dimensione))
        m = len(gruppi)
        for segmento, gruppo in gruppi:
            n = len(gruppo)
            riga = {"dimensione": dimensione, "segmento": segmento, "utenti": n,
                    "guadagno": round(float(gruppo["guadagno_qualita"].mean()), 4),
                    "sovraccarico_lambda0_%": round(float(sovr_ril[gruppo.index].mean()) * 100, 1),
                    "sovraccarico_lambda03_%": round(float(gruppo["sovraccarico"].mean()) * 100, 1)}
            if n >= MIN_UTENTI_IC:
                _, lo, hi = bootstrap_ic(gruppo["guadagno_qualita"].values)
                # Con la correzione per 4 test l'estremo cade molto in coda: con pochi
                # ricampionamenti cambierebbe segno cambiando seme.
                _, lob, hib = bootstrap_ic(gruppo["guadagno_qualita"].values,
                                           n=20000, alpha=0.05 / m)
                ms, los, his = bootstrap_ic(gruppo["diff_sovraccarico"].values)
                riga.update({
                    "ic_guadagno": f"[{lo:+.4f}, {hi:+.4f}]",
                    "significativo": bool(lo > 0),
                    "ic_guadagno_bonferroni": f"[{lob:+.4f}, {hib:+.4f}]",
                    "significativo_bonferroni": bool(lob > 0),
                    "sovraccarico_aggiunto_punti": round(ms, 1),
                    "ic_sovraccarico_aggiunto": f"[{los:+.1f}, {his:+.1f}]",
                })
            else:
                riga.update({"ic_guadagno": f"meno di {MIN_UTENTI_IC} utenti",
                             "significativo": "", "ic_guadagno_bonferroni": "",
                             "significativo_bonferroni": "",
                             "sovraccarico_aggiunto_punti": "", "ic_sovraccarico_aggiunto": ""})
            righe_test.append(riga)
            print(f"  {dimensione:<12} {segmento:<22} n={n:>3}  guadagno {riga['guadagno']:+.4f}  "
                  f"IC {riga['ic_guadagno']}  Bonferroni {riga['ic_guadagno_bonferroni']}  "
                  f"sovraccarico {riga['sovraccarico_lambda0_%']}% -> "
                  f"{riga['sovraccarico_lambda03_%']}%")
    pd.DataFrame(righe_test).to_csv(OUT_TAB / "segmenti_guadagno.csv", index=False)

    print("\nConfronti FRA segmenti (utenti diversi: bootstrap a due campioni)")
    righe_conf = []
    for descrizione, colonna, dimensione, a, b in CONFRONTI:
        va = df.loc[df[dimensione] == a, colonna].values
        vb = df.loc[df[dimensione] == b, colonna].values
        scala = 100 if colonna == "sovraccarico" else 1
        md, lo, hi = bootstrap_ic_due_campioni(va * scala, vb * scala)
        cifre = 1 if colonna == "sovraccarico" else 3
        righe_conf.append({"confronto": descrizione, "differenza": round(md, cifre),
                           "ic": f"[{lo:+.{cifre}f}, {hi:+.{cifre}f}]",
                           "significativo": bool(lo > 0 or hi < 0)})
        print(f"  {descrizione:<44} {md:+.{cifre}f} [{lo:+.{cifre}f}, {hi:+.{cifre}f}]")
    pd.DataFrame(righe_conf).to_csv(OUT_TAB / "segmenti_confronti.csv", index=False)

    figura_attrezzatura(df, righe_conf)

    df.to_csv(OUT_TAB / "segmenti_per_utente.csv", index=False)
    print("\nSalvati: outputs/tables/segmenti_*.csv, outputs/figures/segmenti_attrezzatura.png")


def figura_attrezzatura(df, righe_conf):
    """Disegna gradimento, qualita' e sovraccarico per attrezzatura, con gli intervalli.

    Il pannello del sovraccarico riporta il test a due campioni, perche' le sole medie
    fanno vedere differenze che il test non regge.
    """
    applica_stile()
    ordine = ["Dumbbell Only", "At Home", "Garage Gym", "Full Gym"]
    gruppi = {a: df[df["attrezzatura"] == a] for a in ordine}
    # i nomi lunghi vanno a capo, cosi' le etichette non si toccano
    nomi = {a: (NOMI_ATTREZZATURE[a].replace(" ", "\n", 1) if len(NOMI_ATTREZZATURE[a]) > 8
                else NOMI_ATTREZZATURE[a]) for a in ordine}
    etichette = [f"{nomi[a]}\n({len(gruppi[a])} utenti)" for a in ordine]
    x = np.arange(len(ordine))

    def medie_ic(colonna, scala=1.0):
        stime = [bootstrap_ic(gruppi[a][colonna].values * scala) for a in ordine]
        medie = np.array([s[0] for s in stime])
        return medie, [medie - np.array([s[1] for s in stime]),
                       np.array([s[2] for s in stime]) - medie]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    barre_err = {"ecolor": INCHIOSTRO_2, "elinewidth": 1.2, "capsize": 3}

    ax = axes[0]
    m_ril, e_ril = medie_ic("ndcg_rilevanza")
    m_tot, e_tot = medie_ic("ndcg_completo")
    ax.bar(x - 0.2, m_ril, 0.38, yerr=e_ril, color=BLU_CHIARO, hatch="///",
           edgecolor=INCHIOSTRO_2, linewidth=0.6, error_kw=barre_err, label="sola rilevanza")
    ax.bar(x + 0.2, m_tot, 0.38, yerr=e_tot, color=BLU, edgecolor=INCHIOSTRO_2,
           linewidth=0.6, error_kw=barre_err, label="completo (rilevanza + qualità)")
    ax.set_title("Gradimento (NDCG@10)", loc="left")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.2), ncol=1, frameon=False, fontsize=9)

    ax = axes[1]
    m_q, e_q = medie_ic("qualita_top10")
    ax.bar(x, m_q, 0.55, yerr=e_q, color=VERDE, error_kw=barre_err)
    ax.set_title("Qualità media dei programmi consigliati", loc="left")
    ax.set_ylim(0, 1)

    ax = axes[2]
    m_s, e_s = medie_ic("sovraccarico", 100.0)
    ax.bar(x, m_s, 0.55, yerr=e_s, color=ROSSO, error_kw=barre_err)
    ax.set_title("Consigli oltre la capacità (%)", loc="left")
    conf = next(r for r in righe_conf if r["confronto"] == "sovraccarico, solo manubri - palestra")
    lo, hi = [float(v) for v in conf["ic"].strip("[]").replace("+", "").split(",")]
    esito = "significativa" if conf["significativo"] else "non significativa"
    ax.text(0.0, -0.2, f"Solo manubri − palestra completa: {numero(conf['differenza'], 1, segno=True)} "
            f"punti [{numero(lo, 1, segno=True)}; {numero(hi, 1, segno=True)}],\n"
            f"differenza {esito} (bootstrap a due campioni).",
            transform=ax.transAxes, fontsize=9, color=INCHIOSTRO_2, ha="left", va="top")

    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(etichette, fontsize=9)
        virgola(ax, "y")
        pulisci(ax, griglia="y")
    axes[0].set_ylabel("media per utente — barre = intervallo 95%")
    fig.suptitle("Come viene servito chi ha attrezzature diverse", fontsize=13,
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    salva(fig, "segmenti_attrezzatura")


if __name__ == "__main__":
    main()

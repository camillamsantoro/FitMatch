"""01 - Analisi esplorativa del dataset dei programmi di allenamento.

Produce tabelle in outputs/tables e figure in outputs/figures.
Eseguire dalla radice del progetto:  python scripts/01_eda.py
"""
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_loading import load_summary, load_detail, split_reps, OUT_FIG, OUT_TAB
from src.grafici import (applica_stile, numero, virgola, pulisci, salva,
                         NOMI_ATTREZZATURE, BLU, INCHIOSTRO_2)

OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_TAB.mkdir(parents=True, exist_ok=True)


def sezione(titolo):
    print("\n" + "=" * 70)
    print(titolo)
    print("=" * 70)


def salva_tabella(df, nome):
    path = OUT_TAB / f"{nome}.csv"
    df.to_csv(path, index=True)
    print(f"  -> salvata {path.relative_to(OUT_TAB.parents[1])}")


def figura_programmi(prog):
    """Disegna durata, minuti e attrezzatura dei programmi grezzi.

    La durata va per settimane intere, una barra per settimana, cosi' le classi non
    stanno a cavallo di due settimane.
    """
    applica_stile()
    n = len(prog)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    durata = prog["program_length"].dropna().astype(int).value_counts().sort_index()
    settimane = range(1, int(durata.index.max()) + 1)
    axes[0].bar(list(settimane), durata.reindex(settimane, fill_value=0).values,
                width=0.8, color=BLU)
    axes[0].set_xticks([1, 4, 8, 12, 16, int(durata.index.max())])
    axes[0].set_xlabel("settimane (durata dichiarata)")
    axes[0].set_title("Durata", loc="left")

    minuti = prog["time_per_workout"].dropna()
    bordi = list(range(int(minuti.min()) - 5, int(minuti.max()) + 10, 10))
    axes[1].hist(minuti, bins=bordi, color=BLU, edgecolor="white", linewidth=0.5)
    axes[1].set_xlabel("minuti per seduta (classi di 10 minuti)")
    axes[1].set_title("Minuti per allenamento", loc="left")

    att = prog["equipment"].value_counts()
    axes[2].bar([NOMI_ATTREZZATURE.get(a, a) for a in att.index], att.values,
                width=0.6, color=BLU)
    for i, v in enumerate(att.values):
        axes[2].text(i, v, numero(v, 0, migliaia=True), ha="center", va="bottom",
                     fontsize=9, color=INCHIOSTRO_2)
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].set_title("Attrezzatura richiesta", loc="left")

    axes[2].set_ylim(0, att.max() * 1.12)
    # la nota sta sull'asse perche' dentro il riquadro coprirebbe le barre
    senza = {0: ("durata", int(prog["program_length"].isna().sum())),
             1: ("minuti", int(prog["time_per_workout"].isna().sum())),
             2: ("attrezzatura", int(prog["equipment"].isna().sum()))}
    for i, ax in enumerate(axes):
        ax.set_ylabel("programmi")
        virgola(ax, "y")
        pulisci(ax, griglia="y")
        campo, n_senza = senza[i]
        if n_senza:
            esistente = ax.get_xlabel()
            ax.set_xlabel((esistente + "\n" if esistente else "")
                          + f"({n_senza} {'programma' if n_senza == 1 else 'programmi'} "
                          f"senza {campo} dichiarat{'i' if campo == 'minuti' else 'a'}, escluso)")
    fig.suptitle(f"I {numero(n, 0, migliaia=True)} programmi prima della pulizia",
                 fontsize=13, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    salva(fig, "distribuzioni_programmi")


def conta_multi_valore(serie_liste, nome):
    """Conta i singoli valori di un campo con piu' valori (level, goal)."""
    conteggio = Counter()
    for valori in serie_liste:
        conteggio.update(valori)
    tab = pd.Series(conteggio).sort_values(ascending=False).to_frame("programmi")
    print(f"\n{nome} - {len(tab)} valori distinti, "
          f"{serie_liste.apply(len).mean():.2f} valori per programma in media")
    print(tab.to_string())
    salva_tabella(tab, f"distribuzione_{nome}")
    return tab


def main():
    # ------------------------------------------------------------------ 1
    sezione("1. PROGRAMMI: struttura e completezza")
    prog = load_summary()
    print(f"Programmi: {len(prog)} | colonne: {len(prog.columns) - 2}")
    print(f"Titoli unici: {prog['title'].nunique()} (chiave valida: "
          f"{prog['title'].nunique() == len(prog)})")
    mancanti = prog.isna().sum()
    print("\nValori mancanti:\n", mancanti[mancanti > 0].to_string())

    # ------------------------------------------------------------------ 2
    sezione("2. CAMPI DEL PROFILO UTENTE")
    conta_multi_valore(prog["level_list"], "livello")
    conta_multi_valore(prog["goal_list"], "obiettivo")

    tab_eq = prog["equipment"].value_counts(dropna=False).to_frame("programmi")
    print("\nattrezzatura (valore singolo):\n", tab_eq.to_string())
    salva_tabella(tab_eq, "distribuzione_attrezzatura")

    num = prog[["program_length", "time_per_workout", "total_exercises"]].describe()
    print("\nCampi numerici:\n", num.round(2).to_string())
    salva_tabella(num, "statistiche_numeriche_programmi")

    figura_programmi(prog)

    # ------------------------------------------------------------------ 3
    sezione("3. ESERCIZI: struttura del file di dettaglio")
    det = split_reps(load_detail())
    print(f"Righe: {len(det)} | programmi coperti: {det['title'].nunique()}")
    print(f"Programmi del riepilogo senza esercizi: "
          f"{(~prog['title'].isin(set(det['title']))).sum()}")
    print(f"Esercizi distinti: {det['exercise_name'].nunique()}")

    print(f"\nEsercizi a tempo (ripetizioni negative = secondi): "
          f"{det['is_time_based'].mean() * 100:.1f}% delle righe")
    print(f"Durate impossibili (oltre 2 ore): {(det['duration_s'] > 7200).sum()} righe")

    print("\nIntensita = scala RPE 0-10")
    print(det["intensity"].describe().round(2).to_string())

    top_ex = det["exercise_name"].value_counts().head(20).to_frame("occorrenze")
    print("\nEsercizi piu frequenti:\n", top_ex.to_string())
    salva_tabella(top_ex, "esercizi_piu_frequenti")

    # ------------------------------------------------------------------ 4
    sezione("4. INDICATORI STRUTTURALI PER PROGRAMMA")
    g = det.groupby("title")
    feat = pd.DataFrame({
        "righe": g.size(),
        "settimane": g["week"].max(),
        "giorni_settimana": g["day"].max(),
        "esercizi_distinti": g["exercise_name"].nunique(),
        "serie_totali": g["sets"].sum(),
        "rpe_medio": g["intensity"].mean(),
        "quota_a_tempo": g["is_time_based"].mean() * 100,
    })
    feat["serie_settimana"] = feat["serie_totali"] / feat["settimane"]
    feat["esercizi_sessione"] = feat["righe"] / (feat["settimane"] * feat["giorni_settimana"])
    feat["varieta"] = feat["esercizi_distinti"] / feat["righe"]

    colonne = ["settimane", "giorni_settimana", "esercizi_distinti", "serie_settimana",
               "esercizi_sessione", "rpe_medio", "varieta", "quota_a_tempo"]
    print(feat[colonne].describe().round(2).to_string())
    salva_tabella(feat[colonne].describe(), "statistiche_indicatori_programmi")
    feat.to_csv(OUT_TAB / "indicatori_per_programma.csv")

    # ------------------------------------------------------------------ 5
    sezione("5. PROGRESSIONE NEL TEMPO")
    settimane = det.groupby(["title", "week"]).agg(
        serie=("sets", "sum"), rpe=("intensity", "mean")).reset_index()

    def pendenza(dati, colonna):
        if len(dati) < 3:
            return np.nan
        return np.polyfit(dati["week"], dati[colonna], 1)[0]

    prog_vol = settimane.groupby("title").apply(
        lambda d: pendenza(d, "serie"), include_groups=False)
    prog_rpe = settimane.groupby("title").apply(
        lambda d: pendenza(d, "rpe"), include_groups=False)

    def riassumi(serie, nome, soglia):
        crescita = int((serie > soglia).sum())
        calo = int((serie < -soglia).sum())
        piatti = int((serie.abs() <= soglia).sum())
        print(f"\n{nome}: in crescita {crescita} | piatti {piatti} | "
              f"in calo {calo} | non calcolabile {int(serie.isna().sum())}")
        return pd.Series({"in_crescita": crescita, "piatti": piatti,
                          "in_calo": calo, "non_calcolabile": int(serie.isna().sum())},
                         name=nome)

    tab_prog = pd.concat([
        riassumi(prog_vol, "volume (serie per settimana)", 0.5),
        riassumi(prog_rpe, "intensita (RPE medio)", 0.05),
    ], axis=1)
    salva_tabella(tab_prog, "progressione_programmi")

    ripetuti = det.groupby("title").apply(
        lambda d: d.groupby("week")["exercise_name"].apply(
            lambda s: tuple(sorted(s))).nunique() == 1 and d["week"].nunique() > 1,
        include_groups=False)
    print(f"\nProgrammi con gli stessi esercizi in tutte le settimane: "
          f"{int(ripetuti.sum())} ({ripetuti.mean() * 100:.1f}%)")

    # ------------------------------------------------------------------ 6
    sezione("6. ANOMALIE DA GESTIRE NELLA PULIZIA")
    anomalie = {
        "giorni a settimana > 7": int((feat["giorni_settimana"] > 7).sum()),
        "serie a settimana > 200": int((feat["serie_settimana"] > 200).sum()),
        "programmi di 1 sola settimana": int((feat["settimane"] == 1).sum()),
        "programmi senza livello dichiarato": int((prog["level_list"].apply(len) == 0).sum()),
        "programmi senza obiettivo dichiarato": int((prog["goal_list"].apply(len) == 0).sum()),
        "righe con durate impossibili": int((det["duration_s"] > 7200).sum()),
    }
    tab_anom = pd.Series(anomalie).to_frame("conteggio")
    print(tab_anom.to_string())
    salva_tabella(tab_anom, "anomalie")

    print("\nFatto. Tabelle in outputs/tables, figure in outputs/figures.")


if __name__ == "__main__":
    main()

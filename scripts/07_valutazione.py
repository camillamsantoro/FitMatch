"""07 - Confronto fra i sistemi sugli utenti simulati.

Metriche: NDCG@10, Precision@10, utilita' della top-10, copertura e diversita', con
intervallo bootstrap al 95%. Salva outputs/tables/valutazione_*.csv e
outputs/figures/curva_lambda.png
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
import json
from src.data_loading import testo_programmi, impronta_testi, OUT_TAB
from src.grafici import (applica_stile, numero, virgola, pulisci, salva, intervallo,
                         errori, BLU, ROSSO, INCHIOSTRO_2)
from src.simulator import carica_verita
from src.recommenders import costruisci_sistemi, chiave_query
from src.evaluation import (ndcg_at_k, precision_at_k, utilita_media_top_k,
                            copertura_catalogo, diversita_intra_lista, bootstrap_ic,
                            ordina)

PROCESSED = ROOT / "data" / "processed"
K = 10


def valuta(sistemi, catalogo, utenti, U, quote, parametri_sistema=None):
    parametri_sistema = parametri_sistema or {}
    risultati = {nome: {"ndcg": [], "precision": [], "utilita": [], "diversita": [], "liste": []}
                 for nome in sistemi}
    quote_np = quote.reindex(catalogo["title"]).fillna(0).values
    rng = np.random.default_rng(7)

    for i, utente in enumerate(utenti):
        profilo = utente.profilo_dichiarato()
        for nome, sistema in sistemi.items():
            p = np.asarray(sistema(catalogo, profilo, rng=rng,
                                   **parametri_sistema.get(nome, {})), dtype=float)
            p = np.nan_to_num(p)
            top = ordina(p, K)
            risultati[nome]["ndcg"].append(ndcg_at_k(p, U[i], K))
            risultati[nome]["precision"].append(precision_at_k(p, U[i], K))
            risultati[nome]["utilita"].append(utilita_media_top_k(p, U[i], K))
            risultati[nome]["diversita"].append(diversita_intra_lista(top, quote_np))
            risultati[nome]["liste"].append(set(top))

    righe = []
    for nome, dati in risultati.items():
        riga = {"sistema": nome}
        for metrica in ["ndcg", "precision", "utilita", "diversita"]:
            media, basso, alto = bootstrap_ic(dati[metrica])
            riga[metrica] = round(media, 3)
            riga[f"{metrica}_ic"] = f"[{basso:.3f}, {alto:.3f}]"
        riga["copertura"] = round(copertura_catalogo(dati["liste"], len(catalogo)), 3)
        righe.append(riga)
    return pd.DataFrame(righe).set_index("sistema"), risultati


def figura_lambda(curva, ic_lambda, per_utente):
    """Disegna l'NDCG al variare di lambda e il compromesso con la copertura.

    La nota riporta le differenze appaiate con lambda = 0,3, altrimenti il punto piu'
    alto sembra un ottimo anche se non si distingue dai vicini.
    """
    applica_stile()
    lam = list(curva["lambda"])
    ndcg = list(curva["ndcg"])
    ic = [ic_lambda[l] for l in lam]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))

    ax = axes[0]
    ax.plot(lam, ndcg, color=BLU, linewidth=1.2, zorder=1)
    ax.errorbar(lam, ndcg, yerr=errori(ndcg, ic), fmt="o", color=BLU, markersize=6,
                elinewidth=1.3, capsize=3, zorder=2)
    i3 = lam.index(0.3)
    ax.plot([0.3], [ndcg[i3]], marker="o", markersize=11, markerfacecolor="none",
            markeredgecolor=ROSSO, markeredgewidth=1.6, zorder=3)
    ax.annotate("λ = 0,3 (usato)", (0.3, ndcg[i3]), xytext=(0.32, 0.64),
                fontsize=9, color=ROSSO, arrowprops={"arrowstyle": "-", "color": ROSSO})
    ax.set_xlabel("peso della qualità (λ)")
    ax.set_ylabel("NDCG@10 — barre = intervallo 95%")
    ax.set_title("Gradimento al variare del peso della qualità", loc="left")
    virgola(ax, "x", 1)
    virgola(ax, "y", 2)
    pulisci(ax, griglia="both")

    ax = axes[1]
    ax.plot(curva["ndcg"], curva["copertura"], marker="o", color=BLU)
    # la curva torna indietro: etichette dei primi punti a sinistra, degli ultimi a
    # destra, cosi' non finiscono sulla linea
    spostamenti = {0.0: (-8, 0), 0.1: (-8, 0), 0.2: (-8, 0), 0.3: (-8, 0),
                   0.4: (8, 0), 0.5: (8, -2), 0.7: (-8, 0), 1.0: (0, -13)}
    for l, x, c in zip(lam, curva["ndcg"], curva["copertura"]):
        dx, dy = spostamenti.get(l, (8, 0))
        ax.annotate(f"λ = {numero(l, 1)}", (x, c), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8.5, ha="left" if dx > 0 else ("right" if dx < 0 else "center"),
                    va="center",
                    color=INCHIOSTRO_2)
    ax.set_ylim(bottom=float(curva["copertura"].min()) - 0.035)
    ax.set_xlabel("NDCG@10")
    ax.set_ylabel("copertura del catalogo")
    ax.set_title("Compromesso fra gradimento e copertura", loc="left")
    virgola(ax, "x", 2)
    virgola(ax, "y", 2)
    pulisci(ax, griglia="both")

    from src.evaluation import bootstrap_ic
    parti = []
    for altro in (0.4, 0.5):
        m, lo, hi = bootstrap_ic(per_utente[altro] - per_utente[0.3])
        esito = "non si distingue" if lo < 0 < hi else ("è migliore" if lo > 0 else "è peggiore")
        parti.append(f"{numero(altro, 1)} {esito} ({numero(m, 4, segno=True)} "
                     f"[{numero(lo, 4, segno=True)}; {numero(hi, 4, segno=True)}])")
    nota = "Differenze appaiate con λ = 0,3 (seme 42): " + "; ".join(parti) + "."
    fig.text(0.01, -0.02, nota, fontsize=9, color=INCHIOSTRO_2, ha="left", va="top")
    fig.tight_layout()
    salva(fig, "curva_lambda")


# Il profilo dell'esempio "i sette sistemi su una persona sola". Lo calcola 07
# insieme alla valutazione, cosi' resta allineato ai dati.
PROFILO_ESEMPIO = {"obiettivo": "Muscle & Sculpting", "livello": "Beginner",
                   "attrezzatura": "At Home", "giorni_settimana": 2, "minuti_per_seduta": 30}


def esempio(sistemi, catalogo, utenti, U, n_mostrati=3):
    """Restituisce i primi programmi di ogni sistema per il primo utente con PROFILO_ESEMPIO.

    Il sistema casuale consuma il generatore: per riavere la stessa classifica si
    riparte dal seme e si rifanno gli utenti precedenti nello stesso ordine.
    """
    indice = next(i for i, u in enumerate(utenti)
                  if all(getattr(u, k) == v for k, v in PROFILO_ESEMPIO.items()))
    rng = np.random.default_rng(7)
    righe = []
    for i, utente in enumerate(utenti[:indice + 1]):
        profilo = utente.profilo_dichiarato()
        for nome, sistema in sistemi.items():
            p = np.nan_to_num(np.asarray(sistema(catalogo, profilo, rng=rng), dtype=float))
            if i < indice:
                continue
            for posizione, j in enumerate(ordina(p, n_mostrati), start=1):
                righe.append({"utente": indice, "sistema": nome,
                              "ndcg": round(float(ndcg_at_k(p, U[i], K)), 3),
                              "posizione": posizione,
                              "programma": catalogo["title"].iloc[j],
                              "giorni": int(catalogo["giorni_settimana"].iloc[j]),
                              "minuti": int(catalogo["time_per_workout"].iloc[j]),
                              "utilita_vera": round(float(U[i][j]), 2),
                              "qualita": round(float(catalogo["qualita"].iloc[j]), 2)})
    return pd.DataFrame(righe)


def main():
    catalogo = pd.read_csv(PROCESSED / "catalogo_con_qualita.csv")
    quote = pd.read_csv(PROCESSED / "quote_muscolari.csv", index_col=0)
    U, utenti = carica_verita(catalogo, PROCESSED)
    testi = testo_programmi(catalogo)
    # gli embedding si usano solo se 12 li ha gia' calcolati
    embeddings = None
    percorso_emb = PROCESSED / "embeddings_programmi.npy"
    if not percorso_emb.exists():
        # meglio fermarsi che uscire con sei sistemi invece di sette
        print("ATTENZIONE: embedding non trovati, la baseline con sentence-transformers "
              "e' ESCLUSA. Eseguire scripts/12_embeddings.py per i sette sistemi.")
    else:
        # Gli embedding in cache devono venire da questo testo, in quest'ordine, e
        # coprire tutti gli utenti.
        manifesto = json.loads((PROCESSED / "embeddings_manifesto.json").read_text())
        if manifesto["impronta_testi"] != impronta_testi(testi):
            raise RuntimeError("Gli embedding in cache non corrispondono al testo attuale "
                               "dei programmi (testo o ordine cambiati): rieseguire "
                               "scripts/12_embeddings.py.")
        chiavi = pd.read_csv(PROCESSED / "embeddings_query_chiavi.csv")["query"].tolist()
        mancanti = {chiave_query(u.profilo_dichiarato()) for u in utenti} - set(chiavi)
        if mancanti:
            raise RuntimeError(f"Mancano gli embedding di {len(mancanti)} query della "
                               "popolazione attuale: rieseguire scripts/12_embeddings.py.")
        vettori = np.load(PROCESSED / "embeddings_query.npy")
        embeddings = (np.load(percorso_emb), dict(zip(chiavi, vettori)))
        print(f"Embedding verificati ({manifesto['modello']}, revisione "
              f"{manifesto['revisione'][:8]}): inclusa la baseline con sentence-transformers.")
    sistemi = costruisci_sistemi(catalogo, testi, embeddings=embeddings)

    print(f"Utenti: {len(utenti)} | programmi: {len(catalogo)} | K = {K}\n")
    tabella, dettagli = valuta(sistemi, catalogo, utenti, U, quote)

    print("=" * 100)
    print("CONFRONTO TRA I SISTEMI (medie con intervallo di confidenza bootstrap al 95%)")
    print("=" * 100)
    print(tabella[["ndcg", "ndcg_ic", "precision", "precision_ic",
                   "utilita", "utilita_ic", "copertura", "diversita"]].to_string())
    tabella.to_csv(OUT_TAB / "valutazione_sistemi.csv")
    esempio(sistemi, catalogo, utenti, U).to_csv(OUT_TAB / "esempio_utente.csv", index=False)

    # --- curva al variare di lambda ---
    print("\n" + "=" * 60)
    print("PESO DELLA QUALITA' (lambda): curva di compromesso")
    print("=" * 60)
    righe = []
    # solo per la figura: la tabella non ha gli intervalli per utente
    ic_lambda, per_utente = {}, {}
    for lam in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        tab, dettagli_lam = valuta({"completo (rilevanza + qualita)": sistemi["completo (rilevanza + qualita)"]},
                        catalogo, utenti, U, quote,
                        parametri_sistema={"completo (rilevanza + qualita)": {"lambda_qualita": lam}})
        riga = tab.iloc[0]
        ic_lambda[lam] = intervallo(riga["ndcg_ic"])
        per_utente[lam] = np.asarray(dettagli_lam["completo (rilevanza + qualita)"]["ndcg"])
        righe.append({"lambda": lam, "ndcg": riga["ndcg"], "precision": riga["precision"],
                      "utilita": riga["utilita"], "copertura": riga["copertura"],
                      "diversita": riga["diversita"]})
        print(f"  lambda = {lam:.1f}  ->  NDCG@10 {riga['ndcg']:.3f} | "
              f"P@10 {riga['precision']:.3f} | copertura {riga['copertura']:.3f}")
    curva = pd.DataFrame(righe)
    curva.to_csv(OUT_TAB / "valutazione_lambda.csv", index=False)

    figura_lambda(curva, ic_lambda, per_utente)

    # --- confronto diretto fra i due sistemi chiave ---
    print("\n" + "=" * 60)
    print("IL CONTRIBUTO DELLA QUALITA' E' STATISTICAMENTE SIGNIFICATIVO?")
    print("=" * 60)
    a = np.array(dettagli["rilevanza graduale"]["ndcg"], dtype=float)
    b = np.array(dettagli["completo (rilevanza + qualita)"]["ndcg"], dtype=float)
    differenze = b - a
    media, basso, alto = bootstrap_ic(differenze)
    print(f"  differenza media di NDCG@10 (completo - rilevanza): {media:+.4f}")
    print(f"  intervallo di confidenza al 95%: [{basso:+.4f}, {alto:+.4f}]")
    significativo = (basso > 0) or (alto < 0)
    print(f"  L'intervallo {'NON ' if not significativo else ''}esclude lo zero: "
          f"differenza {'significativa' if significativo else 'non significativa'}")
    migliorati = float((differenze > 0).mean() * 100)
    peggiorati = float((differenze < 0).mean() * 100)
    pareggi = float((differenze == 0).mean() * 100)
    print(f"  utenti migliorati: {migliorati:.1f}% | peggiorati: {peggiorati:.1f}% | "
          f"pareggi esatti: {pareggi:.1f}%")
    pd.DataFrame({"differenza_ndcg": differenze}).describe().to_csv(
        OUT_TAB / "confronto_qualita.csv")
    # I pareggi a parte: se ce ne fossero, migliorati + peggiorati non farebbe 100.
    pd.DataFrame([{
        "differenza_media": round(media, 4), "ic": f"[{basso:+.4f}, {alto:+.4f}]",
        "utenti_migliorati_%": round(migliorati, 1), "utenti_peggiorati_%": round(peggiorati, 1),
        "pareggi_%": round(pareggi, 1),
        "primo_quartile": round(float(np.percentile(differenze, 25)), 4),
        "minimo": round(float(differenze.min()), 3), "massimo": round(float(differenze.max()), 3),
    }]).to_csv(OUT_TAB / "confronto_appaiato.csv", index=False)
    np.save(PROCESSED / "differenze_qualita.npy", differenze)

    print("\nSalvati: outputs/tables/valutazione_sistemi.csv, valutazione_lambda.csv")


if __name__ == "__main__":
    main()

"""06 - Generazione degli utenti simulati e controlli di sanita'.

Salva in data/processed i profili (utenti_simulati.csv), le quote muscolari e la
matrice utenti x programmi (utilita.npy), che fa da verita' di riferimento.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loading import OUT_TAB
from src.catalog import ATTREZZATURE
from src.evaluation import ndcg_at_k
from src.simulator import (genera_utenti, utilita, impronta_titoli, probabilita_rifiuto,
                           fattori_utilita)
from src.scoring import rilevanza, punteggio

PROCESSED = ROOT / "data" / "processed"
N_UTENTI = 300


def calcola_quote_muscolari(catalogo):
    """Legge le quote muscolari prodotte da 05_features.py.

    Non le ricalcola: una seconda versione invecchierebbe senza errori. Se il file
    manca, si ferma.
    """
    percorso = PROCESSED / "quote_muscolari.csv"
    if not percorso.exists():
        raise FileNotFoundError(
            f"{percorso} non esiste: eseguire prima 'python scripts/05_features.py', "
            "che e' l'unico punto in cui le quote muscolari vengono calcolate.")
    return pd.read_csv(percorso, index_col=0)


def main():
    catalogo = pd.read_csv(PROCESSED / "catalogo_con_qualita.csv")
    quote = calcola_quote_muscolari(catalogo)
    utenti = genera_utenti(N_UTENTI)

    print(f"Utenti generati: {len(utenti)} | programmi: {len(catalogo)}")

    righe = []
    U = np.zeros((len(utenti), len(catalogo)))
    for i, u in enumerate(utenti):
        U[i] = utilita(catalogo, quote, u).values
        righe.append({
            "utente": i, "obiettivo": u.obiettivo, "livello": u.livello,
            "attrezzatura": u.attrezzatura, "giorni": u.giorni_settimana,
            "minuti": u.minuti_per_seduta,
            "capacita_serie": round(u.capacita_serie, 1),
            "varieta_preferita": round(u.varieta_preferita, 3),
            "rigidita_orario": round(u.rigidita_orario, 2),
            # l'enfasi e' cio' che il sistema riceve: la salvo per poterla controllare
            "enfasi": "+".join(u.profilo_dichiarato().enfasi),
        })
    profili = pd.DataFrame(righe)
    profili.to_csv(PROCESSED / "utenti_simulati.csv", index=False)
    np.save(PROCESSED / "utilita.npy", U)
    # impronta del catalogo, che carica_verita() controlla
    (PROCESSED / "utilita_titoli.sha256").write_text(impronta_titoli(catalogo) + "\n")

    print("\n=== COMPOSIZIONE DELLA POPOLAZIONE ===")
    for colonna in ["livello", "attrezzatura", "obiettivo"]:
        print(f"\n{colonna}:")
        print((profili[colonna].value_counts(normalize=True) * 100).round(1).to_string())

    print("\n=== CONTROLLI DI SANITA' SULL'UTILITA' ===")
    print(f"utilita' media su tutto il catalogo ..... {U.mean():.3f}")
    print(f"programmi con utilita' 0 (non fattibili)  {(U == 0).mean()*100:.1f}%")
    migliori = U.max(axis=1)
    print(f"utilita' del programma migliore: media {migliori.mean():.3f}, minimo {migliori.min():.3f}")
    sopra = (U > 0.8).sum(axis=1)
    print(f"programmi 'molto buoni' (utilita' > 0,8) per utente: "
          f"mediana {np.median(sopra):.0f}, minimo {sopra.min()}, massimo {sopra.max()}")

    profili["programmi_fattibili"] = (U > 0).sum(axis=1)
    profili["molto_buoni"] = sopra
    print("\nProgrammi fattibili per attrezzatura dichiarata:")
    per_attrezzatura = (profili.groupby("attrezzatura")[["programmi_fattibili", "molto_buoni"]]
                        .median().astype(int))
    print(per_attrezzatura.to_string())

    # La soglia 0,8 per "molto buono" serve solo a leggere i dati, non ha una
    # motivazione esterna.
    sanita = [
        {"misura": "quota di programmi non fattibili (%)", "valore": round(float((U == 0).mean() * 100), 1)},
        {"misura": "programmi molto buoni per utente, mediana", "valore": int(np.median(sopra))},
        {"misura": "programmi molto buoni per utente, minimo", "valore": int(sopra.min())},
        {"misura": "programmi molto buoni per utente, massimo", "valore": int(sopra.max())},
    ]
    for attr, r in per_attrezzatura.iterrows():
        sanita.append({"misura": f"molto buoni, mediana, {attr}", "valore": int(r["molto_buoni"])})
        sanita.append({"misura": f"programmi fattibili, mediana, {attr}", "valore": int(r["programmi_fattibili"])})
    pd.DataFrame(sanita).to_csv(OUT_TAB / "sanita_simulatore.csv", index=False)

    # Gli altri script rigenerano gli utenti dal seme: questo CSV serve anche a
    # controllare che vengano uguali.

    print("\n=== IL SIMULATORE E' INDIPENDENTE DAL SISTEMA? ===")
    print("(correlazione tra utilita' nascosta e punteggio del sistema, per utente)")
    correlazioni_ril, correlazioni_tot = [], []
    for i, u in enumerate(utenti):
        p = u.profilo_dichiarato()
        # Non fattibili a 0 e non a -inf: qui serve una correlazione.
        r = rilevanza(catalogo, p).fillna(0)
        t = punteggio(catalogo, p).fillna(0)
        correlazioni_ril.append(np.corrcoef(U[i], r)[0, 1])
        correlazioni_tot.append(np.corrcoef(U[i], t)[0, 1])
    print(f"  solo rilevanza ....... media {np.mean(correlazioni_ril):.3f}")
    print(f"  rilevanza + qualita' . media {np.mean(correlazioni_tot):.3f}")
    print("  Attesa: positiva ma lontana da 1. Se fosse vicina a 1, il simulatore")
    print("  starebbe ripetendo il punteggio e la valutazione sarebbe circolare.")

    rifiuti = np.mean([probabilita_rifiuto(catalogo, u).mean() for u in utenti])
    print(f"\nProbabilita' media di rifiuto su tutto il catalogo: {rifiuti:.2f}")

    profili.describe().to_csv(OUT_TAB / "utenti_simulati_statistiche.csv")
    pd.DataFrame({"correlazione_rilevanza": correlazioni_ril,
                  "correlazione_totale": correlazioni_tot}).describe().to_csv(
        OUT_TAB / "controllo_circolarita.csv")

    # --- dove passa la circolarita', fattore per fattore ---
    # Per ogni fattore nascosto confronto la sua classifica con quella della
    # componente del sistema che guarda la stessa cosa. Uso Spearman perche' le
    # metriche guardano l'ordine, non la forma della funzione.
    from scipy.stats import spearmanr
    from src.scoring import componenti
    from src.quality import RIFERIMENTO_EQUILIBRIO

    print("\n=== DA DOVE PASSA LA CIRCOLARITA' ===")
    print("(concordanza di Spearman fra la classifica del fattore nascosto e quella")
    print(" della componente del sistema che guarda la stessa cosa)")
    coppie = {"obiettivo": "obiettivo", "livello": "livello", "tempo": "tempo",
              "giorni": "giorni", "carico": "sostenibilita", "muscoli": "enfasi"}
    accumulo = {k: [] for k in coppie}
    accumulo["varieta (voto di qualita')"] = []
    accumulo["muscoli (equilibrio della qualita')"] = []
    rif = pd.Series(RIFERIMENTO_EQUILIBRIO)
    quote_rif = quote.reindex(catalogo["title"]).fillna(0)
    vicinanza_rif = -(quote_rif[rif.index] - rif).abs().sum(axis=1).values
    for i, u in enumerate(utenti):
        pr = u.profilo_dichiarato()
        fattibili = (catalogo["attrezzatura_num"] <= ATTREZZATURE[u.attrezzatura]).values
        co = componenti(catalogo, pr)
        fat = fattori_utilita(catalogo, quote, u)
        for nome_f, nome_c in coppie.items():
            x = np.asarray(fat[nome_f][fattibili], dtype=float)
            y = np.asarray(co[nome_c].values[fattibili], dtype=float)
            # Se un vettore e' costante non c'e' una classifica e Spearman non e' definita:
            # lo salto qui invece di avere NaN e un avviso di scipy.
            if np.ptp(x) == 0 or np.ptp(y) == 0:
                continue
            r = spearmanr(x, y).statistic
            if np.isfinite(r):
                accumulo[nome_f].append(r)
        r = spearmanr(fat["varieta"][fattibili],
                      catalogo["q_varieta"].values[fattibili], nan_policy="omit").statistic
        if np.isfinite(r):
            accumulo["varieta (voto di qualita')"].append(r)
        des = pd.Series(u.distribuzione_desiderata)
        vicinanza_utente = -(quote_rif[des.index] - des).abs().sum(axis=1).values
        r = spearmanr(vicinanza_utente[fattibili], vicinanza_rif[fattibili]).statistic
        if np.isfinite(r):
            accumulo["muscoli (equilibrio della qualita')"].append(r)
    righe_circ = []
    for nome, valori in accumulo.items():
        if not valori:
            continue
        righe_circ.append({"fattore": nome, "concordanza_media": round(float(np.mean(valori)), 3),
                           "minimo": round(float(np.min(valori)), 3)})
        print(f"  {nome:38s} {np.mean(valori):+.3f}  (minimo {np.min(valori):+.3f})")
    pd.DataFrame(righe_circ).to_csv(OUT_TAB / "circolarita_per_fattore.csv", index=False)
    print("  Vicino a 1 = il sistema riproduce quella preferenza nascosta.")
    print("  Salvato: outputs/tables/circolarita_per_fattore.csv")

    # --- il tetto: quanto puo' fare un sistema perfetto ---
    # Il rumore non si puo' prevedere, quindi l'NDCG massimo non e' 1. Serve per
    # leggere il guadagno della qualita' sulla scala giusta.
    print("\n=== IL TETTO PER UN SISTEMA PERFETTO ===")
    U0 = np.vstack([utilita(catalogo, quote, u, con_rumore=False).fillna(0).values
                    for u in utenti])
    tetto = float(np.mean([ndcg_at_k(U0[i], U[i], 10) for i in range(len(utenti))]))
    print(f"  NDCG@10 di chi ordina per utilita' SENZA rumore: {tetto:.3f}")
    import inspect
    sigma = inspect.signature(utilita).parameters["sigma_rumore"].default
    print(f"  Il rumore lognormale (sigma {sigma}) e' irriducibile: un sistema che")
    print("  conoscesse tutti i fattori nascosti si fermerebbe li'.")
    # --- quanto pesa il rumore sulle due metriche ---
    # Rifaccio l'utilita' con altre estrazioni del rumore e confronto le top-10 ideali.
    # Le top-10 cambiano molto, l'NDCG poco: per questo la metrica principale e' l'NDCG.
    condivisi, ndcg_altra = [], []
    for r in range(20):
        semi = np.random.default_rng(r).integers(0, 10**6, len(utenti))
        for i, s in enumerate(semi):
            u2 = U0[i] * np.random.default_rng(s).lognormal(0, sigma, len(U0[i]))
            u2 = np.clip(u2 / max(u2.max(), 1e-9), 0, 1)
            condivisi.append(len(set(np.argsort(-U[i])[:10]) & set(np.argsort(-u2)[:10])))
            ndcg_altra.append(ndcg_at_k(u2, U[i], 10))
    print(f"  Con un'altra estrazione del rumore le top-10 ideali hanno in comune "
          f"{np.mean(condivisi):.1f} programmi su 10, e l'NDCG@10 dell'una giudicata con "
          f"l'altra e' {np.mean(ndcg_altra):.3f}")
    pd.DataFrame([
        {"misura": "tetto NDCG@10 (sistema perfetto)", "valore": round(tetto, 3)},
        {"misura": "top-10 ideali in comune fra due estrazioni del rumore (su 10)",
         "valore": round(float(np.mean(condivisi)), 1)},
        {"misura": "NDCG@10 della top-10 ideale di un'altra estrazione del rumore",
         "valore": round(float(np.mean(ndcg_altra)), 3)},
    ]).to_csv(OUT_TAB / "tetto_ndcg.csv", index=False)
    print("  Salvato: outputs/tables/tetto_ndcg.csv")
    print("\nSalvati: utenti_simulati.csv, utilita.npy  (quote_muscolari.csv lo produce 05)")


if __name__ == "__main__":
    main()

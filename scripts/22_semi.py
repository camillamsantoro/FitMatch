"""22 - Quanto i risultati dipendono dai due semi arbitrari della valutazione.

Faccio variare un seme alla volta, spareggio fra pari merito o rumore di gusto, e
riporto NDCG, guadagno, misto - metadato e in quanti scenari restano significativi.
Va dopo 06 e 08. Salva semi_spareggi.csv e semi_rumore.csv.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import src.evaluation as valutazione
from src.data_loading import OUT_TAB
from src.evaluation import ndcg_at_k, bootstrap_ic, trasforma_metadato
from src.recommenders import NON_FATTIBILE
from src.scoring import rilevanza
from src.simulator import genera_utenti, utilita, carica_verita

PROCESSED = ROOT / "data" / "processed"
VALORI = list(range(20))

_spec = importlib.util.spec_from_file_location("robustezza", ROOT / "scripts" / "10_robustezza.py")
robustezza = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(robustezza)
SCENARI, con_colonna, K, N_UTENTI = (robustezza.SCENARI, robustezza.con_colonna,
                                     robustezza.K, robustezza.N_UTENTI)
SEME_SPAREGGIO_PROGETTO = valutazione.SEME_PARI_MERITO


def prepara_scenari(catalogo, quote, qualita, metadato, misto):
    """Prepara per ogni scenario utenti, verita' e punteggi dei sistemi."""
    dati = {}
    for nome, parametri in SCENARI.items():
        parametri = dict(parametri)
        sigma = parametri.pop("sigma_rumore", 0.12)
        seme = parametri.pop("seme", 42)
        if nome == "riferimento":
            U, utenti = carica_verita(catalogo, PROCESSED)
        else:
            utenti = genera_utenti(N_UTENTI, seme=seme, **parametri)
            U = np.array([utilita(catalogo, quote, u, sigma_rumore=sigma).values
                          for u in utenti])
        base = np.array([utilita(catalogo, quote, u, con_rumore=False).values for u in utenti])
        # la verita' col rumore ufficiale deve essere quella di 10
        scarto = np.abs(con_rumore(base, [u.seme for u in utenti], sigma) - U).max()
        if scarto > 1e-12:
            raise RuntimeError(f"scenario '{nome}': la ricostruzione dell'utilita' differisce "
                               f"di {scarto:.1e} da utilita()")
        punteggi = {"completo": [], "rilevanza": [], "metadato": [], "misto": []}
        for utente in utenti:
            r = rilevanza(catalogo, utente.profilo_dichiarato()).values
            punteggi["completo"].append(con_colonna(r, qualita))
            punteggi["rilevanza"].append(np.where(np.isnan(r), NON_FATTIBILE, r))
            punteggi["metadato"].append(con_colonna(r, metadato))
            punteggi["misto"].append(con_colonna(r, misto))
        dati[nome] = {"utenti": utenti, "sigma": sigma, "U": U, "base": base,
                      "punteggi": punteggi}
    return dati


def con_rumore(base, semi, sigma):
    """Applica il rumore di una realizzazione alla parte deterministica dell'utilita'.

    Sono gli stessi passi di utilita(): i non fattibili restano a zero.
    """
    U = np.empty_like(base)
    for i, (b, s) in enumerate(zip(base, semi)):
        u = b * np.random.default_rng(s).lognormal(0, sigma, len(b))
        U[i] = np.clip(u / max(u.max(), 1e-9), 0, 1)
    return U


def misura(dati, verita):
    """Calcola le misure della tabella, data una verita' per scenario."""
    def ndcg(U, lista):
        return np.array([ndcg_at_k(p, U[i], K) for i, p in enumerate(lista)])

    riga = {}
    guadagni_significativi = misti_significativi = 0
    for nome, d in dati.items():
        U, p = verita[nome], d["punteggi"]
        completo, ril = ndcg(U, p["completo"]), ndcg(U, p["rilevanza"])
        g = bootstrap_ic(completo - ril)
        m = bootstrap_ic(ndcg(U, p["misto"]) - ndcg(U, p["metadato"]))
        guadagni_significativi += int(g[1] > 0)
        misti_significativi += int(m[1] > 0)
        if nome == "riferimento":
            riga.update({
                "ndcg_completo": round(float(completo.mean()), 4),
                "ndcg_rilevanza": round(float(ril.mean()), 4),
                "guadagno": round(g[0], 4), "guadagno_ic_basso": round(g[1], 4),
                "guadagno_ic_alto": round(g[2], 4),
                "misto_meno_metadato": round(m[0], 4), "misto_ic_basso": round(m[1], 4),
                "misto_ic_alto": round(m[2], 4)})
    riga["scenari_guadagno_significativo"] = guadagni_significativi
    riga["scenari_misto_significativo"] = misti_significativi
    return riga


def con_riassunto(righe, chiave):
    tab = pd.DataFrame(righe)
    riassunto = {chiave: "minimo / mediana / massimo", "progetto": ""}
    for c in tab.columns:
        if c not in (chiave, "progetto"):
            v = tab[c].astype(float)
            riassunto[c] = f"{v.min():g} / {v.median():g} / {v.max():g}"
    return pd.concat([tab, pd.DataFrame([riassunto])], ignore_index=True)


def stampa(etichetta, riga):
    print(f"  {etichetta:<22} completo {riga['ndcg_completo']:.4f} | rilevanza "
          f"{riga['ndcg_rilevanza']:.4f} | guadagno {riga['guadagno']:+.4f} "
          f"[{riga['guadagno_ic_basso']:+.4f}; {riga['guadagno_ic_alto']:+.4f}] | misto-metadato "
          f"{riga['misto_meno_metadato']:+.4f} [{riga['misto_ic_basso']:+.4f}; "
          f"{riga['misto_ic_alto']:+.4f}] | scenari: guadagno "
          f"{riga['scenari_guadagno_significativo']}/8, misto {riga['scenari_misto_significativo']}/8")


def main():
    catalogo = pd.read_csv(PROCESSED / "catalogo_con_qualita.csv")
    quote = pd.read_csv(PROCESSED / "quote_muscolari.csv", index_col=0)
    scelta = pd.read_csv(OUT_TAB / "proxy_metadati.csv").set_index("sostituto").loc[
        "numero di esercizi"]
    alfa = float(scelta["alfa_misto"])
    metadato = trasforma_metadato(catalogo["total_exercises"].values, scelta["trasformazione"])
    qualita = catalogo["qualita"].fillna(catalogo["qualita"].median()).values
    misto = alfa * qualita + (1 - alfa) * metadato

    print("Utilita' e punteggi per gli 8 scenari...")
    dati = prepara_scenari(catalogo, quote, qualita, metadato, misto)
    ufficiale = {nome: d["U"] for nome, d in dati.items()}

    # --- 1. lo spareggio: cambia la permutazione, la verita' resta quella ufficiale
    print("\n=== SPAREGGIO FRA PARI MERITO ===")
    righe = []
    try:
        for seme in [SEME_SPAREGGIO_PROGETTO] + VALORI:
            valutazione.SEME_PARI_MERITO = seme
            valutazione._permutazioni.clear()  # la permutazione in cache e' quella del seme prima
            riga = {"seme_spareggio": seme, "progetto": seme == SEME_SPAREGGIO_PROGETTO,
                    **misura(dati, ufficiale)}
            righe.append(riga)
            stampa(f"seme {seme}{' (progetto)' if riga['progetto'] else ''}", riga)
    finally:
        valutazione.SEME_PARI_MERITO = SEME_SPAREGGIO_PROGETTO
        valutazione._permutazioni.clear()
    con_riassunto(righe, "seme_spareggio").to_csv(OUT_TAB / "semi_spareggi.csv", index=False)

    # --- 2. il rumore: ogni utente riceve un seme di rumore nuovo, il resto non
    # cambia
    print("\n=== REALIZZAZIONE DEL RUMORE DI GUSTO ===")
    righe = [{"realizzazione": "progetto", "progetto": True, **misura(dati, ufficiale)}]
    stampa("progetto", righe[0])
    for r in VALORI:
        verita = {}
        for nome, d in dati.items():
            semi = np.random.default_rng(r).integers(0, 10**6, len(d["utenti"]))
            verita[nome] = con_rumore(d["base"], semi, d["sigma"])
        riga = {"realizzazione": r, "progetto": False, **misura(dati, verita)}
        righe.append(riga)
        stampa(f"realizzazione {r}", riga)
    con_riassunto(righe, "realizzazione").to_csv(OUT_TAB / "semi_rumore.csv", index=False)

    for nome in ("semi_spareggi", "semi_rumore"):
        s = pd.read_csv(OUT_TAB / f"{nome}.csv").iloc[:-1]
        print(f"\n{nome}: guadagno significativo nello scenario di riferimento in "
              f"{int((s['guadagno_ic_basso'].astype(float) > 0).sum())} casi su {len(s)}; "
              f"misto - metadato in {int((s['misto_ic_basso'].astype(float) > 0).sum())} su {len(s)}")
        for col in ("scenari_guadagno_significativo", "scenari_misto_significativo"):
            conteggi = s[col].astype(int).value_counts().sort_index()
            print(f"  {col}: " + ", ".join(f"{k}/8 in {v} casi" for k, v in conteggi.items()))
    print("\nSalvato: outputs/tables/semi_spareggi.csv, semi_rumore.csv")


if __name__ == "__main__":
    main()

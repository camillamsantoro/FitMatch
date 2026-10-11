"""21 - Taratura di lambda e del peso dell'enfasi su utenti separati.

Sceglierli sugli stessi utenti della valutazione sarebbe tarare sul test. Qui uso
300 utenti con un altro seme: se l'ottimo cade nello stesso punto, la scelta regge.
Salva outputs/tables/taratura_indipendente.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loading import OUT_TAB
from src.simulator import genera_utenti, utilita
from src.scoring import punteggio, PESI_DEFAULT, PESI_ENFASI_PROVATI as ENFASI_PROVATI
from src.recommenders import NON_FATTIBILE
from src.evaluation import ndcg_at_k, bootstrap_ic

PROCESSED = ROOT / "data" / "processed"
K = 10
SEME_TARATURA = 7  # diverso dal 42 della valutazione
LAMBDA_PROVATI = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]


def main():
    catalogo = pd.read_csv(PROCESSED / "catalogo_con_qualita.csv")
    quote = pd.read_csv(PROCESSED / "quote_muscolari.csv", index_col=0)

    utenti = genera_utenti(300, seme=SEME_TARATURA)
    profili = [u.profilo_dichiarato() for u in utenti]
    print(f"Popolazione di taratura: {len(utenti)} utenti, seme {SEME_TARATURA} "
          f"(la valutazione usa il seme 42)")

    U = np.vstack([utilita(catalogo, quote, u).fillna(0).values for u in utenti])
    print(f"Verita' di riferimento ricalcolata: {U.shape[0]} x {U.shape[1]} giudizi")

    righe = []
    per_utente = {}
    print("\n=== PESO DELLA QUALITA' (lambda) ===")
    for lam in LAMBDA_PROVATI:
        nd = np.array([ndcg_at_k(punteggio(catalogo, p, lambda_qualita=lam).fillna(NON_FATTIBILE).values,
                                 U[i], K) for i, p in enumerate(profili)])
        per_utente[("lambda", lam)] = nd
        m, lo, hi = bootstrap_ic(nd)
        righe.append({"parametro": "lambda", "valore": lam, "ndcg": round(m, 3),
                      "ic": f"[{lo:.3f}, {hi:.3f}]"})
        print(f"  lambda = {lam:.1f} -> NDCG@10 {m:.3f}  [{lo:.3f}, {hi:.3f}]")

    print("\n=== PESO DELLA COMPONENTE DI ENFASI ===")
    for peso in ENFASI_PROVATI:
        pesi = dict(PESI_DEFAULT, enfasi=peso)
        nd = np.array([ndcg_at_k(punteggio(catalogo, p, pesi=pesi).fillna(NON_FATTIBILE).values, U[i], K)
                       for i, p in enumerate(profili)])
        per_utente[("peso_enfasi", peso)] = nd
        m, lo, hi = bootstrap_ic(nd)
        righe.append({"parametro": "peso_enfasi", "valore": peso, "ndcg": round(m, 3),
                      "ic": f"[{lo:.3f}, {hi:.3f}]"})
        print(f"  peso {peso:.2f} -> NDCG@10 {m:.3f}  [{lo:.3f}, {hi:.3f}]")

    tab = pd.DataFrame(righe)
    tab.to_csv(OUT_TAB / "taratura_indipendente.csv", index=False)

    # Confronto appaiato: gli utenti sono gli stessi, quindi faccio il bootstrap
    # sulla differenza. Due intervalli che si toccano possono nascondere una
    # differenza netta.
    esiti, righe_diff = [], []
    for par, scelto in [("lambda", 0.30), ("peso_enfasi", PESI_DEFAULT["enfasi"])]:
        sotto = tab[tab["parametro"] == par]
        migliore = float(sotto.loc[sotto["ndcg"].idxmax(), "valore"])
        base = per_utente[(par, scelto)]
        print(f"\n{par}: valore usato {scelto}, ottimo fuori campione {migliore}")
        indistinguibili = []
        for v in sotto["valore"]:
            if np.isclose(v, scelto):
                indistinguibili.append(v)
                continue
            m, lo, hi = bootstrap_ic(per_utente[(par, v)] - base)
            significativa = lo > 0 or hi < 0
            if not significativa:
                indistinguibili.append(v)
            righe_diff.append({"parametro": par, "valore": v, "valore_usato": scelto,
                               "differenza": round(m, 4), "ic": f"[{lo:+.4f}, {hi:+.4f}]",
                               "significativa": bool(significativa)})
            print(f"  {v} - {scelto}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
                  f"{'SIGNIFICATIVA' if significativa else 'non significativa'}")
        print(f"  valori indistinguibili da {scelto}: {indistinguibili}")
        esiti.append({"parametro": par, "valore_usato": scelto,
                      "ottimo_fuori_campione": migliore,
                      "indistinguibili_dal_valore_usato": " ".join(str(v) for v in indistinguibili)})
    pd.DataFrame(righe_diff).to_csv(OUT_TAB / "taratura_differenze.csv", index=False)
    pd.DataFrame(esiti).to_csv(OUT_TAB / "taratura_esito.csv", index=False)
    print("\nSalvato: outputs/tables/taratura_indipendente.csv")


if __name__ == "__main__":
    main()

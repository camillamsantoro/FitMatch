"""13 - Il sistema impara dai rifiuti?

Conta quante proposte servono prima che l'utente ne accetti una. Un terzo braccio
corregge la stima senza imparare, per separare l'apprendimento; poi si toglie il carico
dal rifiuto simulato. Salva adattivo.csv, adattivo_scomposizione.csv e
adattivo_senza_carico.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loading import OUT_TAB
from src.simulator import genera_utenti, probabilita_rifiuto, utilita, SOGLIA_SOVRACCARICO
from src.recommenders import NON_FATTIBILE
from src.scoring import punteggio
from src.adaptive import sessione
from src.evaluation import bootstrap_ic

PROCESSED = ROOT / "data" / "processed"
N_PROPOSTE = 8

# La capacita' stimata dai dati dichiarati puo' sbagliare di molto: e' proprio il
# caso in cui imparare dai rifiuti serve.
SCENARI = {
    "stima corretta": 1.0,
    "utenti piu' fragili del previsto (-40%)": 0.6,
    "utenti piu' resistenti del previsto (+40%)": 1.4,
}
# (nome, adattivo, impara)
BRACCI = [("statico", False, True),
          ("adattivo senza apprendimento", True, False),
          ("adattivo", True, True)]


def simula(catalogo, utenti, U, con_carico=True):
    """Simula le sessioni di ogni braccio e raccoglie i risultati per utente.

    Utilita' e sovraccarico sono NaN se l'utente non accetta niente.
    """
    ris = {nome: {"proposte": [], "accettato": [], "utilita": [], "sovraccarico": []}
           for nome, _, _ in BRACCI}
    for i, utente in enumerate(utenti):
        profilo = utente.profilo_dichiarato()
        base = punteggio(catalogo, profilo).fillna(NON_FATTIBILE).values
        rifiuti = probabilita_rifiuto(catalogo, utente, con_carico=con_carico).values
        for nome, adattivo, impara in BRACCI:
            giri, scelto = sessione(catalogo, profilo, utente, base, rifiuti,
                                    adattivo=adattivo, n_proposte=N_PROPOSTE,
                                    seme=1000 + i, impara=impara)
            ris[nome]["accettato"].append(giri is not None)
            ris[nome]["proposte"].append(giri if giri is not None else N_PROPOSTE)
            if scelto is None:
                ris[nome]["utilita"].append(np.nan)
                ris[nome]["sovraccarico"].append(np.nan)
            else:
                ris[nome]["utilita"].append(float(U[i][scelto]))
                ris[nome]["sovraccarico"].append(float(
                    catalogo["volume_settimanale"].iloc[scelto]
                    > utente.capacita_serie * SOGLIA_SOVRACCARICO))
    return {n: {k: np.array(v, dtype=float) for k, v in d.items()} for n, d in ris.items()}


def fmt(m, lo, hi, cifre=2):
    return f"[{lo:+.{cifre}f}, {hi:+.{cifre}f}]"


def main():
    catalogo = pd.read_csv(PROCESSED / "catalogo_con_qualita.csv")
    quote = pd.read_csv(PROCESSED / "quote_muscolari.csv", index_col=0)
    righe, righe_sc, righe_sk = [], [], []
    for nome_scenario, scala in SCENARI.items():
        print("\n" + "#" * 84)
        print(f"SCENARIO: {nome_scenario}")
        utenti = genera_utenti(300, scala_capacita=scala)
        U = np.array([utilita(catalogo, quote, u).values for u in utenti])
        r = simula(catalogo, utenti, U)
        statico = r["statico"]["proposte"]
        for nome, _, _ in BRACCI:
            d = r[nome]
            mp, lp, hp = bootstrap_ic(d["proposte"])
            mu, lu, hu = bootstrap_ic(d["utilita"])
            ms, ls, hs = bootstrap_ic(statico - d["proposte"])
            righe.append({
                "sistema": nome,
                "accettazione_%": round(float(np.mean(d["accettato"])) * 100, 1),
                "proposte_medie": round(mp, 2),
                "proposte_ic": f"[{lp:.2f}, {hp:.2f}]",
                "utilita_accettato": round(mu, 3),
                "utilita_ic": f"[{lu:.3f}, {hu:.3f}]",
                "accettati_in_sovraccarico_%": round(float(np.nanmean(d["sovraccarico"])) * 100, 1),
                "scenario": nome_scenario,
                "proposte_risparmiate": round(ms, 2) if nome != "statico" else
                    round(float(bootstrap_ic(statico - r["adattivo"]["proposte"])[0]), 2),
                "ic_risparmio": fmt(ms, ls, hs) if nome != "statico" else
                    fmt(*bootstrap_ic(statico - r["adattivo"]["proposte"])),
            })
            print(f"  {nome:<30} proposte {mp:.2f} | utilita' {mu:.3f} | "
                  f"sovraccarico {righe[-1]['accettati_in_sovraccarico_%']:.1f}%")

        # totale = correzione + apprendimento, tutto appaiato
        a, c = r["adattivo"], r["adattivo senza apprendimento"]
        for parte, x, y in [("totale (statico - adattivo)", r["statico"], a),
                            ("correzione (statico - senza apprendimento)", r["statico"], c),
                            ("apprendimento (senza apprendimento - adattivo)", c, a)]:
            mp, lp, hp = bootstrap_ic(x["proposte"] - y["proposte"])
            ds = (x["sovraccarico"] - y["sovraccarico"]) * 100
            msv, lsv, hsv = bootstrap_ic(ds[np.isfinite(ds)])
            du = y["utilita"] - x["utilita"]
            mu, lu, hu = bootstrap_ic(du[np.isfinite(du)])
            righe_sc.append({"scenario": nome_scenario, "parte": parte,
                             "proposte_risparmiate": round(mp, 2), "ic_proposte": fmt(mp, lp, hp),
                             "sovraccarico_evitato_punti": round(msv, 1),
                             "ic_sovraccarico": fmt(msv, lsv, hsv, 1),
                             "utilita_guadagnata": round(mu, 3), "ic_utilita": fmt(mu, lu, hu, 3)})
            print(f"    {parte:<48} proposte {mp:+.2f} {fmt(mp, lp, hp)} | sovraccarico "
                  f"{msv:+.1f} {fmt(msv, lsv, hsv, 1)} | utilita' {mu:+.3f} {fmt(mu, lu, hu, 3)}")

        r0 = simula(catalogo, utenti, U, con_carico=False)
        for parte, x, y in [("totale (statico - adattivo)", r0["statico"], r0["adattivo"]),
                            ("apprendimento", r0["adattivo senza apprendimento"], r0["adattivo"])]:
            mp, lp, hp = bootstrap_ic(x["proposte"] - y["proposte"])
            righe_sk.append({"scenario": nome_scenario, "parte": parte,
                             "proposte_risparmiate": round(mp, 2), "ic_proposte": fmt(mp, lp, hp),
                             "sovraccarico_statico_%": round(float(np.nanmean(x["sovraccarico"])) * 100, 1),
                             "sovraccarico_adattivo_%": round(float(np.nanmean(y["sovraccarico"])) * 100, 1)})
        print(f"    senza carico nel rifiuto: risparmio {righe_sk[-2]['proposte_risparmiate']:+.2f} "
              f"{righe_sk[-2]['ic_proposte']}, apprendimento {righe_sk[-1]['proposte_risparmiate']:+.2f} "
              f"{righe_sk[-1]['ic_proposte']}")

    pd.DataFrame(righe).to_csv(OUT_TAB / "adattivo.csv", index=False)
    pd.DataFrame(righe_sc).to_csv(OUT_TAB / "adattivo_scomposizione.csv", index=False)
    pd.DataFrame(righe_sk).to_csv(OUT_TAB / "adattivo_senza_carico.csv", index=False)
    print("\nSalvato: outputs/tables/adattivo.csv, adattivo_scomposizione.csv, "
          "adattivo_senza_carico.csv")


if __name__ == "__main__":
    main()

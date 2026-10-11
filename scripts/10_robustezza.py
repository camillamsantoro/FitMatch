"""10 - Prova di robustezza: le conclusioni reggono se cambio il simulatore?

Per ogni scenario rigenero utenti e verita', poi misuro in modo appaiato il guadagno
della qualita', l'ordine dei metodi e il contributo sopra un metadato. Trasformazione
e alfa restano quelli scelti da 08: si prova il sistema fissato, non uno ritarato.
Salva robustezza.csv, robustezza_metadato.csv e robustezza_meccanismo.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loading import OUT_TAB, testo_programmi
from src.simulator import genera_utenti, utilita, SOGLIA_SOVRACCARICO
from src.recommenders import costruisci_sistemi, NON_FATTIBILE
from src.scoring import rilevanza, punteggio
from src.evaluation import ndcg_at_k, bootstrap_ic, ordina, trasforma_metadato

PROCESSED = ROOT / "data" / "processed"
K = 10
N_UTENTI = 300
LAMBDA = 0.30

SCENARI = {
    "riferimento": {},
    "capacita' -30%": {"scala_capacita": 0.7},
    "capacita' +30%": {"scala_capacita": 1.3},
    "rumore alto": {"sigma_rumore": 0.30},
    "orari rigidi": {"scala_rigidita": 1.8},
    "piu' utenti a casa": {"distribuzione_attrezzatura": {
        "Full Gym": 0.20, "Garage Gym": 0.20, "At Home": 0.40, "Dumbbell Only": 0.20}},
    "seme diverso": {"seme": 2026},
    "alta variabilita' personale": {"scala_variabilita": 2.0},
}
ORDINE = ["completo (rilevanza + qualita)", "rilevanza graduale", "solo metadati", "casuale"]


def con_colonna(r, colonna, lam=LAMBDA):
    """Fa lo stesso calcolo di punteggio(), con un'altra colonna al posto della qualita'."""
    p = (1 - lam) * r + lam * colonna
    return np.where(np.isnan(p), NON_FATTIBILE, p)


def fmt(lo, hi, cifre=4):
    return f"[{lo:+.{cifre}f}, {hi:+.{cifre}f}]"


def main():
    catalogo = pd.read_csv(PROCESSED / "catalogo_con_qualita.csv")
    quote = pd.read_csv(PROCESSED / "quote_muscolari.csv", index_col=0)
    sistemi = costruisci_sistemi(catalogo, testo_programmi(catalogo))

    # la scelta di 08 sul seme 7: per questo 10 va dopo 08
    if not (OUT_TAB / "proxy_metadati.csv").exists():
        raise FileNotFoundError("manca outputs/tables/proxy_metadati.csv: eseguire prima "
                                "08_ablation.py, che sceglie trasformazione e alfa sul seme 7")
    scelta = pd.read_csv(OUT_TAB / "proxy_metadati.csv").set_index("sostituto").loc[
        "numero di esercizi"]
    alfa = float(scelta["alfa_misto"])
    metadato = trasforma_metadato(catalogo["total_exercises"].values, scelta["trasformazione"])
    qualita = catalogo["qualita"].fillna(catalogo["qualita"].median()).values
    misto = alfa * qualita + (1 - alfa) * metadato
    print(f"Metadato: numero di esercizi, {scelta['trasformazione']}; misto con alfa {alfa}\n")

    righe, righe_meta, righe_mecc = [], [], []
    for nome_scenario, parametri in SCENARI.items():
        parametri = dict(parametri)
        sigma = parametri.pop("sigma_rumore", 0.12)
        seme = parametri.pop("seme", 42)
        utenti = genera_utenti(N_UTENTI, seme=seme, **parametri)
        U = np.array([utilita(catalogo, quote, u, sigma_rumore=sigma).values for u in utenti])

        nd = {}
        rng = np.random.default_rng(7)
        for nome, sistema in sistemi.items():
            valori = []
            for i, utente in enumerate(utenti):
                p = np.nan_to_num(np.asarray(
                    sistema(catalogo, utente.profilo_dichiarato(), rng=rng), dtype=float))
                valori.append(ndcg_at_k(p, U[i], K))
            nd[nome] = np.array(valori, dtype=float)

        nd_meta, nd_misto, stima_su_vera, sovr0, sovr3 = [], [], [], [], []
        for i, utente in enumerate(utenti):
            profilo = utente.profilo_dichiarato()
            r = rilevanza(catalogo, profilo).values
            if i == 0:  # se scoring.py cambia formula, ci si ferma
                atteso = punteggio(catalogo, profilo, lambda_qualita=LAMBDA).fillna(
                    NON_FATTIBILE).values
                if not np.array_equal(con_colonna(r, qualita), atteso):
                    raise RuntimeError("con_colonna non coincide con punteggio()")
            nd_meta.append(ndcg_at_k(con_colonna(r, metadato), U[i], K))
            nd_misto.append(ndcg_at_k(con_colonna(r, misto), U[i], K))
            stima_su_vera.append(profilo.capacita_stimata / utente.capacita_serie)
            volume = catalogo["volume_settimanale"].values
            for lista, p in ((sovr0, np.where(np.isnan(r), NON_FATTIBILE, r)),
                             (sovr3, con_colonna(r, qualita))):
                top = ordina(p, K)
                lista.append(float((volume[top] / utente.capacita_serie
                                    > SOGLIA_SOVRACCARICO).mean()) * 100)
        nd_meta, nd_misto = np.array(nd_meta), np.array(nd_misto)

        riga = {"scenario": nome_scenario}
        for nome, valori in nd.items():
            riga[nome] = round(float(np.nanmean(valori)), 3)
        media, basso, alto = bootstrap_ic(nd[ORDINE[0]] - nd[ORDINE[1]])
        riga["guadagno qualita'"] = round(media, 3)
        riga["ic guadagno"] = f"[{basso:+.3f}, {alto:+.3f}]"
        medie = [float(np.nanmean(nd[s])) for s in ORDINE]
        riga["ordine delle medie"] = all(a > b for a, b in zip(medie, medie[1:]))
        gradini = [bootstrap_ic(nd[a] - nd[b]) for a, b in zip(ORDINE, ORDINE[1:])]
        riga["ordine significativo"] = all(lo > 0 for _, lo, _ in gradini)
        m_rm, lo_rm, hi_rm = gradini[1]
        riga["rilevanza - metadati"] = round(m_rm, 4)
        riga["ic rilevanza - metadati"] = fmt(lo_rm, hi_rm)
        righe.append(riga)

        mq, lq, hq = bootstrap_ic(nd[ORDINE[0]] - nd_meta)
        mm, lm, hm = bootstrap_ic(nd_misto - nd_meta)
        righe_meta.append({
            "scenario": nome_scenario,
            "qualita_meno_metadato": round(mq, 4), "ic_qualita_meno_metadato": fmt(lq, hq),
            "misto_meno_metadato": round(mm, 4), "ic_misto_meno_metadato": fmt(lm, hm),
            "misto_significativo": bool(lm > 0),
        })
        righe_mecc.append({
            "scenario": nome_scenario,
            "stima_su_vera_mediana": round(float(np.median(stima_su_vera)), 2),
            # la mediana non basta: conta la quota di chi viene sovrastimato
            "utenti_sovrastimati_%": round(float(np.mean(np.array(stima_su_vera) > 1)) * 100, 1),
            "sovraccarico_lambda0_%": round(float(np.mean(sovr0)), 1),
            "sovraccarico_lambda03_%": round(float(np.mean(sovr3)), 1),
        })
        print(f"{nome_scenario:<28} guadagno {media:+.4f} {fmt(basso, alto)} | "
              f"ril.-met. {m_rm:+.4f} {fmt(lo_rm, hi_rm)} | ordine "
              f"{'significativo' if riga['ordine significativo'] else 'NON significativo'} | "
              f"misto-met. {mm:+.4f} {fmt(lm, hm)} | stima/vera "
              f"{righe_mecc[-1]['stima_su_vera_mediana']:.2f}")

    tab = pd.DataFrame(righe).set_index("scenario")
    tab.to_csv(OUT_TAB / "robustezza.csv")
    meta = pd.DataFrame(righe_meta).set_index("scenario")
    meta.to_csv(OUT_TAB / "robustezza_metadato.csv")
    pd.DataFrame(righe_mecc).set_index("scenario").to_csv(OUT_TAB / "robustezza_meccanismo.csv")
    print("\n" + "=" * 70)
    print(f"Guadagno della qualita' significativo: "
          f"{sum(1 for v in tab['ic guadagno'] if v.startswith('[+'))} scenari su {len(tab)}")
    print(f"Ordine delle medie invariato: {int(tab['ordine delle medie'].sum())} su {len(tab)}; "
          f"ogni gradino significativo: {int(tab['ordine significativo'].sum())} su {len(tab)}")
    print(f"Misto - metadato significativo: {int(meta['misto_significativo'].sum())} su {len(meta)}")
    print("Salvato: outputs/tables/robustezza.csv, robustezza_metadato.csv, "
          "robustezza_meccanismo.csv")


if __name__ == "__main__":
    main()

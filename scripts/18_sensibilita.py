"""18 - Sensibilita' alle soglie della qualita'.

Le soglie di src/quality.py sono scelte di dominio: le sposto una alla volta e rimisuro
il guadagno sugli stessi utenti. La variante migliore non la scelgo, vorrebbe dire
adattare la qualita' al simulatore. Salva sensibilita_qualita.csv,
sensibilita_seme7.csv e sensibilita_pesi.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loading import load_detail, OUT_TAB
from src.catalog import pulisci_dettaglio
from src.evaluation import ndcg_at_k, bootstrap_ic
from src.quality import (FASCE_RIPETIZIONI, FREQUENZA_IDEALE, PENALITA_COMPONENTE,
                         PESI_QUALITA, PROGRESSIONE_PIENA, RIFERIMENTO_EQUILIBRIO,
                         SOGLIA_COMPONENTE, VARIETA_IDEALE, VOLUME_CONSIGLIATO,
                         _plateau, TOLLERANZA_FASCIA, PENDENZA_NULLA,
                         SETTIMANE_MINIME_PROGRESSIONE, sotto_soglia)
from src.recommenders import NON_FATTIBILE
from src.scoring import punteggio, rilevanza, LAMBDA_QUALITA
from src.simulator import (genera_utenti, carica_verita, utilita, SOGLIA_SOVRACCARICO,
                           impronta_titoli)
from src.evaluation import ordina

PROCESSED = ROOT / "data" / "processed"
ISTOGRAMMA = PROCESSED / "istogramma_ripetizioni.csv"
ISTOGRAMMA_IMPRONTA = PROCESSED / "istogramma_ripetizioni.sha256"
K = 10
LAMBDA = LAMBDA_QUALITA
SEME_TARATURA = 7  # lo stesso di 21

COMPONENTI = ["q_volume", "q_frequenza", "q_progressione",
              "q_equilibrio", "q_varieta", "q_coerenza"]


# --- cache ---
def istogramma_ripetizioni(catalogo) -> pd.DataFrame:
    """Conta le serie per programma e numero di ripetizioni, con una cache.

    La cache si riusa solo se l'impronta dei titoli coincide, cioe' se e' stata fatta
    per questo catalogo.
    """
    impronta = impronta_titoli(catalogo)
    if (ISTOGRAMMA.exists() and ISTOGRAMMA_IMPRONTA.exists()
            and ISTOGRAMMA_IMPRONTA.read_text().strip() == impronta):
        return pd.read_csv(ISTOGRAMMA)
    det, _ = pulisci_dettaglio(load_detail())
    det = det[det["reps"] > 0]
    ist = det.groupby(["title", "reps"])["sets"].sum().reset_index()
    ist.to_csv(ISTOGRAMMA, index=False)
    ISTOGRAMMA_IMPRONTA.write_text(impronta + "\n")
    return ist


# --- ricalcolo degli indicatori ---
def ricalcola_qualita(catalogo, ist, volume=None, frequenza=None, progressione=None,
                      equilibrio=None, varieta=None, fasce=None,
                      soglia=None, penalita=None, pesi=None, tolleranza=None,
                      pendenza_nulla=None, settimane_minime=None, componenti=False):
    """Rifa' i sei indicatori e la qualita' con riferimenti diversi.

    Parte dai valori grezzi del catalogo. Con i riferimenti predefiniti deve dare
    esattamente la qualita' salvata, e main() lo controlla.
    """
    # 'is None' e non 'or', come in scoring.py: un dizionario vuoto non deve
    # diventare il predefinito senza avviso.
    volume = VOLUME_CONSIGLIATO if volume is None else volume
    frequenza = FREQUENZA_IDEALE if frequenza is None else frequenza
    progressione = PROGRESSIONE_PIENA if progressione is None else progressione
    equilibrio = RIFERIMENTO_EQUILIBRIO if equilibrio is None else equilibrio
    varieta = VARIETA_IDEALE if varieta is None else varieta
    fasce = FASCE_RIPETIZIONI if fasce is None else fasce
    settimane_minime = (SETTIMANE_MINIME_PROGRESSIONE if settimane_minime is None
                        else settimane_minime)
    soglia = SOGLIA_COMPONENTE if soglia is None else soglia
    tolleranza = TOLLERANZA_FASCIA if tolleranza is None else tolleranza
    pendenza_nulla = PENDENZA_NULLA if pendenza_nulla is None else pendenza_nulla
    penalita = PENALITA_COMPONENTE if penalita is None else penalita
    pesi = PESI_QUALITA if pesi is None else pesi

    q = pd.DataFrame(index=catalogo.index)

    livello = catalogo["livello_min"].fillna(2).astype(int)
    minimi = livello.map(lambda l: volume[l][0])
    massimi = livello.map(lambda l: volume[l][1])
    q["q_volume"] = [float(_plateau(v, mn, mx, tolleranza)) if np.isfinite(v) else np.nan
                     for v, mn, mx in zip(catalogo["volume_settimanale"], minimi, massimi)]

    q["q_frequenza"] = (catalogo["frequenza_media"] / frequenza).clip(0, 1)

    q["q_progressione"] = ((catalogo["pendenza_rpe"] - pendenza_nulla)
                           / (progressione - pendenza_nulla)).clip(0, 1)
    q.loc[catalogo["settimane"] < settimane_minime, "q_progressione"] = np.nan

    rif = pd.Series(equilibrio)
    quote = catalogo[[f"quota_{g}" for g in rif.index]].copy()
    quote.columns = rif.index
    distanza = (quote - rif).abs().sum(axis=1) / 2
    q["q_equilibrio"] = (1 - distanza).clip(0, 1)
    # Per i programmi senza muscoli le quote sono mancanti e la somma da' 0, quindi
    # l'equilibrio diventa NaN e il peso passa alle altre componenti. E' quello che
    # serve.
    q.loc[quote.sum(axis=1) <= 0, "q_equilibrio"] = np.nan

    q["q_varieta"] = [float(_plateau(v, *varieta, tolleranza)) for v in catalogo["varieta"]]

    colonne_goal = {o: "goal_" + o.lower().replace(" & ", "_").replace(" ", "_")
                    for o in fasce}
    limiti = {}
    for titolo, riga in catalogo.set_index("title").iterrows():
        obiettivi = [o for o, c in colonne_goal.items() if riga.get(c, 0) == 1]
        if obiettivi:
            limiti[titolo] = (min(fasce[o][0] for o in obiettivi),
                              max(fasce[o][1] for o in obiettivi))
    i = ist.copy()
    i["min_fascia"] = i["title"].map(lambda t: limiti.get(t, (np.nan, np.nan))[0])
    i["max_fascia"] = i["title"].map(lambda t: limiti.get(t, (np.nan, np.nan))[1])
    dentro = (i["reps"] >= i["min_fascia"]) & (i["reps"] <= i["max_fascia"])
    coerenza = (i.assign(dentro=dentro * i["sets"]).groupby("title")["dentro"].sum() /
                i.groupby("title")["sets"].sum())
    q["q_coerenza"] = catalogo["title"].map(coerenza).values

    # il peso delle componenti non calcolabili passa alle altre
    serie_pesi = pd.Series({f"q_{k}": v for k, v in pesi.items()})[COMPONENTI]
    valori = q[COMPONENTI]
    pesi_validi = valori.notna() * serie_pesi
    somma = (valori.fillna(0) * serie_pesi).sum(axis=1) / pesi_validi.sum(axis=1)
    sotto = sotto_soglia(valori, soglia).sum(axis=1)
    totale = (somma - penalita * sotto).clip(0, 1)
    return (totale, q) if componenti else totale


# --- varianti ---
def scala_fascia(fasce_volume, fattore):
    return {l: (a * fattore, b * fattore) for l, (a, b) in fasce_volume.items()}


def sposta_equilibrio(rif, gruppo, quanto):
    """Sposta 'quanto' di quota su un gruppo, togliendola agli altri in proporzione."""
    nuovo = dict(rif)
    altri = {g: v for g, v in rif.items() if g != gruppo}
    totale_altri = sum(altri.values())
    nuovo[gruppo] = rif[gruppo] + quanto
    for g, v in altri.items():
        nuovo[g] = v - quanto * v / totale_altri
    return nuovo


def allarga_fasce(fasce, quanto):
    return {o: (max(1, a - quanto), b + quanto) for o, (a, b) in fasce.items()}


def varianti():
    return [
        ("fascia di volume -20%", dict(volume=scala_fascia(VOLUME_CONSIGLIATO, 0.8))),
        ("fascia di volume +20%", dict(volume=scala_fascia(VOLUME_CONSIGLIATO, 1.2))),
        ("frequenza ideale 1,5 invece di 2", dict(frequenza=1.5)),
        ("frequenza ideale 2,5 invece di 2", dict(frequenza=2.5)),
        ("progressione piena a +0,05 RPE", dict(progressione=0.05)),
        ("progressione piena a +0,20 RPE", dict(progressione=0.20)),
        ("equilibrio spostato sulle gambe (+0,10)", dict(equilibrio=sposta_equilibrio(RIFERIMENTO_EQUILIBRIO, "gambe", 0.10))),
        ("equilibrio spostato sulla parte alta (-0,10 gambe)", dict(equilibrio=sposta_equilibrio(RIFERIMENTO_EQUILIBRIO, "gambe", -0.10))),
        ("varieta' ideale 3-40% invece di 5-30%", dict(varieta=(0.03, 0.40))),
        ("varieta' ideale 8-22% invece di 5-30%", dict(varieta=(0.08, 0.22))),
        ("fasce di ripetizioni piu' larghe (+/-3)", dict(fasce=allarga_fasce(FASCE_RIPETIZIONI, 3))),
        ("fasce di ripetizioni piu' strette (-1)", dict(fasce=allarga_fasce(FASCE_RIPETIZIONI, -1))),
        ("soglia di penalita' 0,15 invece di 0,25", dict(soglia=0.15)),
        ("soglia di penalita' 0,35 invece di 0,25", dict(soglia=0.35)),
        ("penalita' 0,05 invece di 0,10", dict(penalita=0.05)),
        ("penalita' 0,20 invece di 0,10", dict(penalita=0.20)),
        ("tolleranza della fascia 0,25 invece di 0,5", dict(tolleranza=0.25)),
        ("tolleranza della fascia 1,0 invece di 0,5", dict(tolleranza=1.0)),
        # Con -0,05 un RPE piatto prende un voto intermedio invece di zero.
        ("RPE piatto non azzerato (origine -0,05)", dict(pendenza_nulla=-0.05)),
        ("pesi tutti uguali", dict(pesi={k: 1 / 6 for k in PESI_QUALITA})),
        # Le settimane minime si possono solo alzare: sotto le tre la pendenza nel
        # catalogo manca gia'.
        ("progressione misurabile da 4 settimane invece di 3", dict(settimane_minime=4)),
        # I pesi uno alla volta li prova gia' l'ablation di 08.
    ]


# --- valutazione ---
_RILEVANZA = {}


def rilevanze(catalogo, utenti):
    """Calcola la rilevanza per utente una volta sola: non dipende dalla qualita'."""
    if id(utenti) not in _RILEVANZA or _RILEVANZA[id(utenti)][0] is not utenti:
        _RILEVANZA[id(utenti)] = (utenti, [rilevanza(catalogo, u.profilo_dichiarato()).values
                                           for u in utenti])
    return _RILEVANZA[id(utenti)][1]


def misura(catalogo, utenti, U, qualita):
    """Calcola NDCG e sovraccarico per utente, come punteggio()."""
    q = qualita.fillna(qualita.median()).values
    volume = catalogo["volume_settimanale"].values
    ndcg, sovr = [], []
    for i, r in enumerate(rilevanze(catalogo, utenti)):
        p = (1 - LAMBDA) * r + LAMBDA * q
        p = np.where(np.isnan(p), NON_FATTIBILE, p)
        if i == 0:
            cat = catalogo.copy()
            cat["qualita"] = qualita.values
            atteso = punteggio(cat, utenti[0].profilo_dichiarato(),
                               lambda_qualita=LAMBDA).fillna(NON_FATTIBILE).values
            if not np.array_equal(p, atteso):
                raise RuntimeError("misura() non coincide con punteggio()")
        ndcg.append(ndcg_at_k(p, U[i], K))
        top = ordina(p, K)
        sovr.append(float((volume[top] / utenti[i].capacita_serie
                           > SOGLIA_SOVRACCARICO).mean()) * 100)
    return np.array(ndcg, dtype=float), np.array(sovr, dtype=float)


def controllo_equivalenza(catalogo, ist):
    """Controlla che il ricalcolo riproduca la qualita' salvata e ogni componente.

    Se no, si ferma. Serve anche a scoprire una cache dell'istogramma vecchia.
    """
    totale, comp = ricalcola_qualita(catalogo, ist, componenti=True)
    # prima le componenti: cosi' l'errore dice quale componente non torna
    confronti = [(c, comp[c], catalogo[c]) for c in COMPONENTI] + [
        ("qualita", totale, catalogo["qualita"])]
    peggiore = 0.0
    for nome, nuovo, salvato in confronti:
        if not nuovo.isna().equals(salvato.isna()):
            raise RuntimeError(f"controllo di equivalenza: i mancanti di '{nome}' non "
                               f"coincidono ({int(nuovo.isna().sum())} contro "
                               f"{int(salvato.isna().sum())})")
        scarto = float((nuovo - salvato).abs().max())
        peggiore = max(peggiore, scarto)
        if scarto > 1e-12:
            raise RuntimeError(f"controllo di equivalenza: '{nome}' differisce di {scarto:.2e}")
    print(f"Controllo di equivalenza superato: qualita' e sei componenti riprodotte, "
          f"mancanti negli stessi programmi, scarto massimo {peggiore:.1e}\n")
    return totale


def main():
    catalogo = pd.read_csv(PROCESSED / "catalogo_con_qualita.csv")
    U, utenti = carica_verita(catalogo, PROCESSED)
    ist = istogramma_ripetizioni(catalogo)
    riprodotta = controllo_equivalenza(catalogo, ist)

    quote = pd.read_csv(PROCESSED / "quote_muscolari.csv", index_col=0)
    utenti7 = genera_utenti(300, seme=SEME_TARATURA)
    U7 = np.vstack([utilita(catalogo, quote, u).fillna(0).values for u in utenti7])

    base_ndcg = np.array([ndcg_at_k(rilevanza(catalogo, u.profilo_dichiarato()).fillna(NON_FATTIBILE).values,
                                    U[i], K) for i, u in enumerate(utenti)], dtype=float)
    rif_ndcg, rif_sovr = misura(catalogo, utenti, U, riprodotta)
    rif7, _ = misura(catalogo, utenti7, U7, riprodotta)

    configurazioni = [("riferimento (soglie attuali)", {})] + varianti()
    n_test = len(configurazioni)
    print("=" * 110)
    print("SENSIBILITA' ALLE SOGLIE DELLA QUALITA'")
    print("=" * 110)
    print(f"{'variante':<52}{'NDCG':>7}{'guadagno':>10}  {'IC 95%':<20}"
          f"{'diff. dal rif.':>15}  {'sovracc.':>8}")
    righe, righe7 = [], []
    for nome, override in configurazioni:
        q = riprodotta if not override else ricalcola_qualita(catalogo, ist, **override)
        ndcg, sovr = misura(catalogo, utenti, U, q)
        ndcg7, _ = misura(catalogo, utenti7, U7, q)
        _, basso, alto = bootstrap_ic(ndcg - base_ndcg)
        _, basso_b, alto_b = bootstrap_ic(ndcg - base_ndcg, alpha=0.05 / n_test)
        riga = {"variante": nome,
                "ndcg": round(float(np.nanmean(ndcg)), 3),
                "guadagno": round(float(np.nanmean(ndcg - base_ndcg)), 4),
                "ic_guadagno": f"[{basso:+.4f}, {alto:+.4f}]",
                "guadagno_significativo": bool(basso > 0),
                "ic_guadagno_bonferroni": f"[{basso_b:+.4f}, {alto_b:+.4f}]",
                "significativo_bonferroni": bool(basso_b > 0),
                "sovraccarico_%": round(float(sovr.mean()), 1)}
        if override:
            md, ld, hd = bootstrap_ic(ndcg - rif_ndcg)
            ms, ls, hs = bootstrap_ic(sovr - rif_sovr)
            riga.update({"differenza_dal_riferimento": round(md, 4),
                         "ic_differenza": f"[{ld:+.4f}, {hd:+.4f}]",
                         "differenza_significativa": bool(ld > 0 or hd < 0),
                         "differenza_sovraccarico_punti": round(ms, 1),
                         "ic_differenza_sovraccarico": f"[{ls:+.1f}, {hs:+.1f}]"})
            m7, l7, h7 = bootstrap_ic(ndcg7 - rif7)
        else:
            riga.update({"differenza_dal_riferimento": "", "ic_differenza": "",
                         "differenza_significativa": "", "differenza_sovraccarico_punti": "",
                         "ic_differenza_sovraccarico": ""})
            m7, l7, h7 = 0.0, np.nan, np.nan
        righe.append(riga)
        righe7.append({"variante": nome, "ndcg_seme7": round(float(ndcg7.mean()), 4),
                       "differenza_seme7": round(m7, 4),
                       "ic_differenza_seme7": "" if not override else f"[{l7:+.4f}, {h7:+.4f}]"})
        print(f"{nome:<52}{riga['ndcg']:>7.3f}{riga['guadagno']:>+10.4f}  "
              f"{riga['ic_guadagno']:<20}{str(riga['differenza_dal_riferimento']):>15}  "
              f"{riga['sovraccarico_%']:>7.1f}%")

    tab = pd.DataFrame(righe)
    tab.to_csv(OUT_TAB / "sensibilita_qualita.csv", index=False)
    tab7 = pd.DataFrame(righe7)
    tab7.to_csv(OUT_TAB / "sensibilita_seme7.csv", index=False)

    n_ok = int(tab["guadagno_significativo"].sum())
    n_bonf = int(tab["significativo_bonferroni"].sum())
    varie = tab.iloc[1:]
    print("-" * 110)
    print(f"Guadagno significativo in {n_ok}/{len(tab)} configurazioni "
          f"({n_bonf}/{len(tab)} con Bonferroni per {n_test} test).")
    print(f"Entita' diversa dal riferimento (appaiato): "
          f"{int(varie['differenza_significativa'].astype(bool).sum())} varianti su {len(varie)}")
    print(f"Guadagno minimo:  {varie['guadagno'].min():+.4f}  "
          f"({varie.sort_values('guadagno').iloc[0]['variante']})")
    print(f"Guadagno massimo: {varie['guadagno'].max():+.4f}  "
          f"({varie.sort_values('guadagno').iloc[-1]['variante']})")
    print(f"Sovraccarico: da {tab['sovraccarico_%'].min():.1f}% a {tab['sovraccarico_%'].max():.1f}%")
    scelta = tab7.sort_values("ndcg_seme7").iloc[-1]
    print(f"\nSul seme {SEME_TARATURA} vincerebbe: {scelta['variante']} "
          f"(NDCG {scelta['ndcg_seme7']:.4f}, {scelta['differenza_seme7']:+.4f} "
          f"{scelta['ic_differenza_seme7']}). NON adottata: vedi il docstring.")

    # Sposto un peso alla volta verso 1/6, per capire da dove viene il vantaggio
    # dei pesi uguali.
    scomposizione = [
        ("pesi tutti uguali", {k: 1 / 6 for k in PESI_QUALITA}),
        ("solo il peso della varieta' a 1/6", {**PESI_QUALITA, "varieta": 1 / 6}),
        ("solo il peso della progressione a 1/6", {**PESI_QUALITA, "progressione": 1 / 6}),
    ]
    righe_pesi = []
    for nome, pesi in scomposizione:
        q = ricalcola_qualita(catalogo, ist, pesi=pesi)
        n42, _ = misura(catalogo, utenti, U, q)
        n7, _ = misura(catalogo, utenti7, U7, q)
        m42, l42, h42 = bootstrap_ic(n42 - rif_ndcg)
        m7, l7, h7 = bootstrap_ic(n7 - rif7)
        righe_pesi.append({"variante": nome, "differenza_seme42": round(m42, 4),
                           "ic_seme42": f"[{l42:+.4f}, {h42:+.4f}]",
                           "differenza_seme7": round(m7, 4), "ic_seme7": f"[{l7:+.4f}, {h7:+.4f}]"})
        print(f"  {nome:<40} seme 42 {m42:+.4f} [{l42:+.4f}, {h42:+.4f}]   "
              f"seme 7 {m7:+.4f} [{l7:+.4f}, {h7:+.4f}]")
    pd.DataFrame(righe_pesi).to_csv(OUT_TAB / "sensibilita_pesi.csv", index=False)
    print("\nSalvato: outputs/tables/sensibilita_qualita.csv, sensibilita_seme7.csv, "
          "sensibilita_pesi.csv")


if __name__ == "__main__":
    main()

"""08 - Ablation sulle componenti di qualita'.

Il guadagno viene da una componente sola o dall'insieme? E la qualita' e' solo un
metadato travestito? Ogni differenza e' variante meno riferimento: positiva vuol dire
che la variante va meglio. Salva ablation_qualita.csv, proxy_metadati.csv e
proxy_trasformazioni_seme7.csv in outputs/tables.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loading import OUT_TAB
from src.simulator import genera_utenti, carica_verita, utilita
from src.quality import PESI_QUALITA, aggrega_qualita
from src.recommenders import NON_FATTIBILE
from src.scoring import punteggio, rilevanza, LAMBDA_QUALITA
from src.evaluation import (ndcg_at_k, precision_at_k, bootstrap_ic,
                            TRASFORMAZIONI_METADATO)

PROCESSED = ROOT / "data" / "processed"
K = 10
LAMBDA = LAMBDA_QUALITA
SEME_TARATURA = 7  # popolazione di taratura, la stessa di 21
ALFA_PROVATI = [0.0, 0.25, 0.5, 0.75, 1.0]


# La rilevanza non cambia fra le varianti, quindi la calcolo una volta per utente.
# Tengo la lista degli utenti nella cache perche' il suo id non venga riusato da
# un'altra popolazione.
_RILEVANZA = {}


def _rilevanze(catalogo, utenti):
    impronta = int(pd.util.hash_pandas_object(
        catalogo.drop(columns="qualita"), index=True).sum())
    chiave = (id(utenti), impronta)
    if chiave not in _RILEVANZA or _RILEVANZA[chiave][0] is not utenti:
        _RILEVANZA[chiave] = (utenti, [rilevanza(catalogo, u.profilo_dichiarato()).values
                                       for u in utenti])
    return _RILEVANZA[chiave][1]


def valuta_catalogo(catalogo, utenti, U, lam=LAMBDA):
    """Fa lo stesso calcolo di punteggio(), con la rilevanza in cache.

    La prima volta controlla che il risultato coincida con punteggio(): se la formula
    cambiasse, lo script si ferma.
    """
    rilevanze = _rilevanze(catalogo, utenti)
    q = catalogo["qualita"].fillna(catalogo["qualita"].median()).values
    ndcg, prec = [], []
    for i, r in enumerate(rilevanze):
        p = (1 - lam) * r + lam * q
        p = np.where(np.isnan(p), NON_FATTIBILE, p)
        if i == 0:
            atteso = punteggio(catalogo, utenti[0].profilo_dichiarato(),
                               lambda_qualita=lam).fillna(NON_FATTIBILE).values
            if not np.array_equal(p, atteso):
                raise RuntimeError("valuta_catalogo non coincide con punteggio()")
        ndcg.append(ndcg_at_k(p, U[i], K))
        prec.append(precision_at_k(p, U[i], K))
    return np.array(ndcg, dtype=float), np.array(prec, dtype=float)


def con_qualita(catalogo, colonna):
    cat = catalogo.copy()
    cat["qualita"] = np.asarray(colonna, dtype=float)
    return cat


def fmt_ic(basso, alto):
    return f"[{basso:+.4f}, {alto:+.4f}]"


def main():
    catalogo = pd.read_csv(PROCESSED / "catalogo_con_qualita.csv")
    U, utenti = carica_verita(catalogo, PROCESSED)

    base_ndcg = []
    for i, utente in enumerate(utenti):
        p = rilevanza(catalogo, utente.profilo_dichiarato()).fillna(NON_FATTIBILE).values
        base_ndcg.append(ndcg_at_k(p, U[i], K))
    base_ndcg = np.array(base_ndcg, dtype=float)

    completo_ndcg, completo_prec = valuta_catalogo(
        con_qualita(catalogo, aggrega_qualita(catalogo, PESI_QUALITA)), utenti, U)

    righe = [{
        "configurazione": "sistema completo",
        "ndcg": round(float(np.nanmean(completo_ndcg)), 3),
        "precision": round(float(np.nanmean(completo_prec)), 3),
        "differenza_ndcg": 0.0,
        "ic_differenza": "",
        "significativo": "",
        "ndcg_mancanti_alla_media": "",
    }]

    def riga(nome, ndcg, prec, extra=""):
        d = ndcg - completo_ndcg
        m, lo, hi = bootstrap_ic(d)
        righe.append({
            "configurazione": nome,
            "ndcg": round(float(np.nanmean(ndcg)), 3),
            "precision": round(float(np.nanmean(prec)), 3),
            "differenza_ndcg": round(m, 4),
            "ic_differenza": fmt_ic(lo, hi),
            "significativo": bool(lo > 0 or hi < 0),
            "ndcg_mancanti_alla_media": extra,
        })
        return m, lo, hi

    print("=" * 88)
    print("ABLATION: cosa succede togliendo una componente alla volta")
    print("(differenza = variante - completo; positiva = la variante va meglio)")
    print("=" * 88)
    print(f"sistema senza qualita' (riferimento)  NDCG@10 = {np.nanmean(base_ndcg):.3f}")
    print(f"sistema completo                      NDCG@10 = {np.nanmean(completo_ndcg):.3f}\n")

    for componente in PESI_QUALITA:
        pesi = {k: (0.0 if k == componente else v) for k, v in PESI_QUALITA.items()}
        ndcg, prec = valuta_catalogo(
            con_qualita(catalogo, aggrega_qualita(catalogo, pesi)), utenti, U)
        m, lo, hi = riga(f"senza {componente}", ndcg, prec)
        print(f"  senza {componente:<14} NDCG@10 = {np.nanmean(ndcg):.3f}   "
              f"differenza {m:+.4f} {fmt_ic(lo, hi)}"
              f"{'' if (lo > 0 or hi < 0) else '  non significativa'}")

    # Con una sola componente, i programmi dove non si calcola prendono la mediana,
    # che per la progressione e' il voto peggiore. Per questo riporto anche il
    # riempimento con la media.
    print("\nTenendo una sola componente:")
    solo_ndcg = {}  # serve anche piu' sotto, nel confronto con la sola rilevanza
    for componente in PESI_QUALITA:
        pesi = {k: (1.0 if k == componente else 0.0) for k in PESI_QUALITA}
        q = aggrega_qualita(catalogo, pesi)
        ndcg, prec = valuta_catalogo(con_qualita(catalogo, q), utenti, U)
        solo_ndcg[componente] = ndcg
        extra = ""
        if q.isna().any():
            ndcg_media, _ = valuta_catalogo(con_qualita(catalogo, q.fillna(q.mean())), utenti, U)
            extra = round(float(np.nanmean(ndcg_media)), 3)
        m, lo, hi = riga(f"solo {componente}", ndcg, prec, extra)
        nota = ""
        if q.isna().any():
            nota = (f"   ({int(q.isna().sum())} mancanti, mediana {q.median():.2f}; "
                    f"con la media {extra:.3f})")
        print(f"  solo {componente:<15} NDCG@10 = {np.nanmean(ndcg):.3f}   "
              f"differenza {m:+.4f} {fmt_ic(lo, hi)}{nota}")

    pd.DataFrame(righe).to_csv(OUT_TAB / "ablation_qualita.csv", index=False)
    print("\nSalvato: outputs/tables/ablation_qualita.csv")

    # --- la qualita' e' solo un metadato travestito? ---
    # Se si', un metadato dichiarato al posto della qualita' darebbe lo stesso guadagno.
    # Trasformazione e verso del metadato si scelgono sul seme 7 e si applicano una
    # volta sul seme 42. Due test: la qualita' batte il metadato, e aggiunge qualcosa?
    print("\n" + "=" * 88)
    print("LA QUALITA' E' SOLO UN TRAVESTIMENTO DEI METADATI?")
    print("=" * 88)

    trasformazioni = TRASFORMAZIONI_METADATO
    obiettivi = [c for c in catalogo.columns if c.startswith("goal_")]
    metadati = {
        "durata in settimane": catalogo["settimane"].values,
        "numero di esercizi": catalogo["total_exercises"].values,
        "numero di obiettivi dichiarati": catalogo[obiettivi].sum(axis=1).values,
    }
    qualita = catalogo["qualita"].fillna(catalogo["qualita"].median()).values

    quote = pd.read_csv(PROCESSED / "quote_muscolari.csv", index_col=0)
    utenti7 = genera_utenti(300, seme=SEME_TARATURA)
    U7 = np.vstack([utilita(catalogo, quote, u).fillna(0).values for u in utenti7])
    print(f"Scelte fatte sul seme {SEME_TARATURA} ({len(utenti7)} utenti), "
          f"risultati sul seme 42.\n")

    righe_seme7, scelte = [], {}
    for nome, valori in metadati.items():
        migliore = None
        for nome_t, f in trasformazioni.items():
            for verso in ("crescente", "decrescente"):
                colonna = f(valori) if verso == "crescente" else 1 - f(valori)
                n7, _ = valuta_catalogo(con_qualita(catalogo, colonna), utenti7, U7)
                righe_seme7.append({"metadato": nome, "trasformazione": nome_t,
                                    "verso": verso, "parametro": "",
                                    "ndcg_seme7": round(float(n7.mean()), 4)})
                if migliore is None or n7.mean() > migliore[1]:
                    migliore = (nome_t, n7.mean(), verso, colonna)
        colonna = migliore[3]
        etichetta = migliore[0] + ("" if migliore[2] == "crescente" else ", decrescente")
        migliore_alfa = None
        for alfa in ALFA_PROVATI:
            n7, _ = valuta_catalogo(
                con_qualita(catalogo, alfa * qualita + (1 - alfa) * colonna), utenti7, U7)
            righe_seme7.append({"metadato": nome, "trasformazione": etichetta,
                                "verso": migliore[2], "parametro": f"alfa={alfa}",
                                "ndcg_seme7": round(float(n7.mean()), 4)})
            if migliore_alfa is None or n7.mean() > migliore_alfa[1]:
                migliore_alfa = (alfa, n7.mean())
        scelte[nome] = (etichetta, colonna, migliore_alfa[0])
        print(f"  seme 7: {nome:<32} trasformazione {etichetta:<14} "
              f"alfa del misto {migliore_alfa[0]}")
    pd.DataFrame(righe_seme7).to_csv(OUT_TAB / "proxy_trasformazioni_seme7.csv", index=False)

    qual_ndcg, _ = valuta_catalogo(con_qualita(catalogo, qualita), utenti, U)
    varieta = catalogo["q_varieta"].fillna(catalogo["q_varieta"].median()).values
    var_ndcg, _ = valuta_catalogo(con_qualita(catalogo, varieta), utenti, U)

    righe_proxy = []

    def aggiungi(nome, trasf, ndcg, confronti):
        m, lo, hi = bootstrap_ic(ndcg - base_ndcg)
        r = {"sostituto": nome, "trasformazione": trasf,
             "ndcg": round(float(ndcg.mean()), 3),
             "guadagno_su_sola_rilevanza": round(m, 4),
             "ic_guadagno": fmt_ic(lo, hi), "significativo": bool(lo > 0),
             "qualita_meno_sostituto": "", "ic_qualita_meno_sostituto": "",
             "alfa_misto": "", "misto_meno_sostituto": "", "ic_misto_meno_sostituto": ""}
        r.update(confronti)
        righe_proxy.append(r)
        print(f"\n  {nome:<32} NDCG {ndcg.mean():.3f} | sulla sola rilevanza {m:+.4f} "
              f"{fmt_ic(lo, hi)}")

    aggiungi("qualita (il nostro punteggio)", "", qual_ndcg, {})
    aggiungi("solo il voto di varieta'", "", var_ndcg, {})
    # La coerenza da sola, confrontata con la sola rilevanza: dice se il suo legame
    # col numero di obiettivi puo' gonfiare il guadagno.
    aggiungi("solo il voto di coerenza", "", solo_ndcg["coerenza"], {})
    for nome, (trasf, colonna, alfa) in scelte.items():
        meta_ndcg, _ = valuta_catalogo(con_qualita(catalogo, colonna), utenti, U)
        misto_ndcg, _ = valuta_catalogo(
            con_qualita(catalogo, alfa * qualita + (1 - alfa) * colonna), utenti, U)
        mq, lq, hq = bootstrap_ic(qual_ndcg - meta_ndcg)
        mm_, lm, hm = bootstrap_ic(misto_ndcg - meta_ndcg)
        aggiungi(nome, trasf, meta_ndcg, {
            "qualita_meno_sostituto": round(mq, 4),
            "ic_qualita_meno_sostituto": fmt_ic(lq, hq),
            "alfa_misto": alfa,
            "misto_meno_sostituto": round(mm_, 4),
            "ic_misto_meno_sostituto": fmt_ic(lm, hm),
        })
        print(f"      qualita' - metadato  {mq:+.4f} {fmt_ic(lq, hq)}"
              f"{'' if lq > 0 else '  NON significativo: la qualita non lo batte'}")
        print(f"      misto (alfa {alfa}) - metadato  {mm_:+.4f} {fmt_ic(lm, hm)}"
              f"{'  la qualita aggiunge informazione' if lm > 0 else '  NON significativo'}")

    tab = pd.DataFrame(righe_proxy)
    tab.to_csv(OUT_TAB / "proxy_metadati.csv", index=False)
    print(f"\n  Correlazioni misurate: varieta' grezza ~ settimane = "
          f"{catalogo['varieta'].corr(catalogo['settimane']):+.3f}, "
          f"voto di varieta' ~ settimane = "
          f"{catalogo['q_varieta'].corr(catalogo['settimane']):+.3f}, "
          f"voto di varieta' ~ log(numero di esercizi) = "
          f"{catalogo['q_varieta'].corr(np.log(catalogo['total_exercises'])):+.3f}")
    print("  Salvato: outputs/tables/proxy_metadati.csv, proxy_trasformazioni_seme7.csv")


if __name__ == "__main__":
    main()

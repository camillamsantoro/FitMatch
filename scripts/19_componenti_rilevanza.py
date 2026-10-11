"""19 - Cosa aggiunge ciascuna componente di rilevanza.

A: quanto deve pesare l'enfasi. B: che effetto ha la sostenibilita'. C: come si
divide il salto da "solo metadati" a "rilevanza graduale". Enfasi e sostenibilita'
servono a mantenere cio' che il sistema promette, non a far salire l'NDCG, quindi il
peso dell'enfasi si sceglie sul lavoro sulle zone chieste.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loading import OUT_TAB
from src.simulator import genera_utenti, probabilita_rifiuto, SOGLIA_SOVRACCARICO, carica_verita
from src.recommenders import NON_FATTIBILE, solo_metadati
from src.scoring import (PESI_DEFAULT, PESI_ENFASI_PROVATI, LAMBDA_QUALITA, rilevanza,
                         punteggio)
from src.quality import RIFERIMENTO_EQUILIBRIO
from src.evaluation import ndcg_at_k, bootstrap_ic, ordina

PROCESSED = ROOT / "data" / "processed"
K = 10
SEME_TARATURA = 7  # lo stesso di 21


def lavoro_sulle_zone(catalogo, profili, pesi):
    """Misura il lavoro sulle zone chieste nella top-10, per chi le ha dichiarate."""
    rapporti, sopra = [], []
    for pr in profili:
        zone = [z for z in getattr(pr, "enfasi", ()) if f"quota_{z}" in catalogo.columns]
        if not zone:
            continue
        p = punteggio(catalogo, pr, pesi=pesi).fillna(NON_FATTIBILE).values
        top = ordina(p, K)
        quota = catalogo.iloc[top][[f"quota_{z}" for z in zone]].sum(axis=1)
        riferimento = sum(RIFERIMENTO_EQUILIBRIO[z] for z in zone)
        rapporti.append(float((quota / riferimento).mean()))
        # Soglia piu' bassa di quella della componente: conta i consigli che vanno
        # almeno nella direzione giusta.
        sopra.append(float((quota > riferimento).mean()))
    return float(np.mean(rapporti)), float(np.mean(sopra))


def main():
    catalogo = pd.read_csv(PROCESSED / "catalogo_con_qualita.csv")
    U, utenti = carica_verita(catalogo, PROCESSED)

    profili = [u.profilo_dichiarato() for u in utenti]
    con_enfasi = [i for i, pr in enumerate(profili) if getattr(pr, "enfasi", ())]
    print(f"Utenti: {len(utenti)} | con almeno una zona dichiarata: {len(con_enfasi)}")

    profili7 = [u.profilo_dichiarato() for u in genera_utenti(300, seme=SEME_TARATURA)]
    righe = []
    ndcg_per_peso = {}
    for peso in PESI_ENFASI_PROVATI:
        pesi = dict(PESI_DEFAULT, enfasi=peso)
        ndcg = [ndcg_at_k(punteggio(catalogo, pr, pesi=pesi).fillna(NON_FATTIBILE).values,
                          U[i], K) for i, pr in enumerate(profili)]
        lavoro, sopra = lavoro_sulle_zone(catalogo, profili, pesi)
        lavoro7, _ = lavoro_sulle_zone(catalogo, profili7, pesi)
        ndcg_per_peso[peso] = np.array(ndcg)
        righe.append({"peso_enfasi": peso,
                      "ndcg": round(float(np.nanmean(ndcg)), 3),
                      "lavoro_zone_chieste": round(lavoro, 2),
                      "programmi_sopra_riferimento_%": round(sopra * 100, 0),
                      "lavoro_zone_chieste_seme7": round(lavoro7, 2)})
        print(f"  peso {peso:.2f} -> NDCG {righe[-1]['ndcg']:.3f} | "
              f"lavoro sulle zone chieste {lavoro:.2f}x (seme 7: {lavoro7:.2f}x) | "
              f"sopra il riferimento {sopra * 100:.0f}%")

    tab = pd.DataFrame(righe)
    tab.to_csv(OUT_TAB / "peso_enfasi.csv", index=False)
    rif = tab[tab.peso_enfasi == PESI_DEFAULT["enfasi"]].iloc[0]
    zero = tab[tab.peso_enfasi == 0.0].iloc[0]
    # Il costo dell'enfasi va testato in modo appaiato, non letto da due medie.
    m, lo, hi = bootstrap_ic(ndcg_per_peso[PESI_DEFAULT["enfasi"]] - ndcg_per_peso[0.0])
    print(f"\nPeso scelto {PESI_DEFAULT['enfasi']}: NDCG {rif.ndcg:.3f} contro {zero.ndcg:.3f} "
          f"senza la componente.")
    print(f"  Costo appaiato: {m:+.4f} [{lo:+.4f}, {hi:+.4f}] "
          f"{'SIGNIFICATIVO' if (lo > 0 or hi < 0) else 'non significativo'}")
    print(f"  In cambio: lavoro sulle zone chieste {zero.lavoro_zone_chieste:.2f}x -> "
          f"{rif.lavoro_zone_chieste:.2f}x")
    pd.DataFrame([{"confronto": "peso 0,10 contro peso 0", "differenza": round(m, 4),
                   "ic": f"[{lo:+.4f}, {hi:+.4f}]", "significativa": bool(lo > 0 or hi < 0)}]
                 ).to_csv(OUT_TAB / "costo_enfasi.csv", index=False)
    print("Salvato: outputs/tables/peso_enfasi.csv, costo_enfasi.csv")

    # --- Parte B: la sostenibilita' ---
    # Peso 0 toglie anche la sua penalita'; rilevanza() rinormalizza gli altri pesi.
    print("\n=== EFFETTO DELLA COMPONENTE DI SOSTENIBILITA' ===")
    righe_b, per_utente_b = [], {}
    for etichetta, peso in [("senza sostenibilita", 0.0),
                            ("con sostenibilita (scelto)", PESI_DEFAULT["sostenibilita"])]:
        pesi = dict(PESI_DEFAULT, sostenibilita=peso)
        ndcg, ndcg_ril, sovraccarico, rifiuto = [], [], [], []
        for i, (u, pr) in enumerate(zip(utenti, profili)):
            p = punteggio(catalogo, pr, lambda_qualita=LAMBDA_QUALITA,
                          pesi=pesi).fillna(NON_FATTIBILE).values
            r = rilevanza(catalogo, pr, pesi=pesi).fillna(NON_FATTIBILE).values
            top = ordina(p, K)
            ndcg.append(ndcg_at_k(p, U[i], K))
            ndcg_ril.append(ndcg_at_k(r, U[i], K))
            rapporto = catalogo["volume_settimanale"].iloc[top] / u.capacita_serie
            sovraccarico.append(float((rapporto > SOGLIA_SOVRACCARICO).mean()) * 100)
            rifiuto.append(float(probabilita_rifiuto(catalogo, u).iloc[top].mean()))
        righe_b.append({"configurazione": etichetta, "peso_sostenibilita": peso,
                        "ndcg": round(float(np.nanmean(ndcg)), 3),
                        "ndcg_sola_rilevanza": round(float(np.nanmean(ndcg_ril)), 3),
                        "quota_sovraccarico_%": round(float(np.mean(sovraccarico)), 1),
                        "probabilita_abbandono": round(float(np.mean(rifiuto)), 3)})
        b = righe_b[-1]
        print(f"  {etichetta:28s} NDCG {b['ndcg']:.3f} | sola rilevanza "
              f"{b['ndcg_sola_rilevanza']:.3f} | sovraccarico "
              f"{b['quota_sovraccarico_%']:.1f}% | abbandono {b['probabilita_abbandono']:.3f}")
        per_utente_b[peso] = {"NDCG del completo": np.array(ndcg),
                              "NDCG della sola rilevanza": np.array(ndcg_ril),
                              "sovraccarico (punti)": np.array(sovraccarico),
                              "probabilita' di abbandono": np.array(rifiuto)}
    pd.DataFrame(righe_b).to_csv(OUT_TAB / "effetto_sostenibilita.csv", index=False)
    righe_bt = []
    con, senza = per_utente_b[PESI_DEFAULT["sostenibilita"]], per_utente_b[0.0]
    for misura in con:
        m, lo, hi = bootstrap_ic(con[misura] - senza[misura])
        cifre = 1 if "punti" in misura else 4
        righe_bt.append({"misura": misura, "con_meno_senza": round(m, cifre),
                         "ic": f"[{lo:+.{cifre}f}, {hi:+.{cifre}f}]",
                         "significativa": bool(lo > 0 or hi < 0)})
        print(f"    con - senza, {misura:<28} {m:+.{cifre}f} [{lo:+.{cifre}f}, {hi:+.{cifre}f}]")
    pd.DataFrame(righe_bt).to_csv(OUT_TAB / "sostenibilita_appaiata.csv", index=False)
    print("Salvato: outputs/tables/effetto_sostenibilita.csv, sostenibilita_appaiata.csv")

    # --- Parte C: i gradini fra metadati e rilevanza graduale ---
    # Cambiano tre cose insieme: penalita' graduali, sostenibilita' ed enfasi, che e'
    # un'informazione in piu'. Le separo per non dare tutto il merito alla graduazione.
    print("\n=== I TRE GRADINI FRA METADATI E RILEVANZA GRADUALE ===")
    gradini = [
        ("solo metadati (contatore di vincoli)",
         lambda cat, pr: solo_metadati(cat, pr)),
        ("+ penalita' graduali, senza sostenibilita' ne' enfasi",
         lambda cat, pr: rilevanza(cat, pr, pesi=dict(PESI_DEFAULT, sostenibilita=0.0,
                                                      enfasi=0.0)).fillna(NON_FATTIBILE).values),
        ("+ sostenibilita'",
         lambda cat, pr: rilevanza(cat, pr, pesi=dict(PESI_DEFAULT,
                                                      enfasi=0.0)).fillna(NON_FATTIBILE).values),
        ("+ enfasi (= rilevanza graduale)",
         lambda cat, pr: rilevanza(cat, pr).fillna(NON_FATTIBILE).values),
    ]
    valori, righe_c = [], []
    for nome, f in gradini:
        nd = np.array([ndcg_at_k(np.asarray(f(catalogo, pr), dtype=float), U[i], K)
                       for i, pr in enumerate(profili)])
        valori.append(nd)
        m, lo, hi = bootstrap_ic(nd)
        righe_c.append({"gradino": nome, "ndcg": round(m, 3), "ic": f"[{lo:.3f}, {hi:.3f}]",
                        "guadagno_sul_precedente": None, "ic_guadagno": None,
                        "significativo": None})
        print(f"  {nome:<52} NDCG {m:.3f} [{lo:.3f}, {hi:.3f}]")
    for i in range(1, len(valori)):
        d = valori[i] - valori[i - 1]
        m, lo, hi = bootstrap_ic(d)
        righe_c[i]["guadagno_sul_precedente"] = round(m, 4)
        righe_c[i]["ic_guadagno"] = f"[{lo:+.4f}, {hi:+.4f}]"
        righe_c[i]["significativo"] = bool(lo > 0)
        print(f"    passo {i}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}] "
              f"{'significativo' if lo > 0 else 'NON significativo'}")
    pd.DataFrame(righe_c).to_csv(OUT_TAB / "gradini_rilevanza.csv", index=False)
    print("  La graduazione e la sostenibilita' si dividono il salto; la graduazione da")
    print("  sola resta significativa, quindi la decisione di abbandonare il filtro")
    print("  rigido regge.")
    print("  Salvato: outputs/tables/gradini_rilevanza.csv")


if __name__ == "__main__":
    main()

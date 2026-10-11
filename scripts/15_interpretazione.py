"""15 - Testo libero: regole contro classificatori addestrati.

Confronta src/richiesta.py con Logistic Regression, Random Forest e SVM su TF-IDF,
misurando tutti sulle frasi di test. Alcune parole scattano solo sul test, quindi
riporto anche il risultato senza di loro. Salva le tabelle interpretazione_*.csv.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_loading import OUT_TAB
from src.evaluation import bootstrap_ic, intervallo_wilson
import src.richiesta as richiesta
from src.richiesta import interpreta

FRASI = ROOT / "data" / "valutazione" / "frasi_utente.csv"
ASSENTE = "(assente)"
CAMPI = ["obiettivo", "livello", "attrezzatura"]


def classificatori():
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42,
                                                class_weight="balanced"),
        "SVM (kernel lineare)": SVC(kernel="linear", class_weight="balanced", random_state=42),
    }


def accuratezza_regole(dati, giusti=None):
    """Calcola l'accuratezza per campo.

    Se 'giusti' e' un dizionario, ci salva per ogni campo l'esito frase per frase.
    """
    previsioni = [interpreta(f) for f in dati["frase"]]
    riga = {}
    for campo in CAMPI:
        atteso = dati[campo].replace("", ASSENTE).values
        predetto = [p.get(campo, ASSENTE) for p in previsioni]
        esatti = atteso == pd.Series(predetto).values
        riga[campo] = round(float(esatti.mean()), 3)
        if giusti is not None:
            giusti[campo] = esatti
    return riga, previsioni


def giusti_modello(modello, sviluppo, test):
    giusti = {}
    for campo in CAMPI:
        pipeline = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), modello)
        pipeline.fit(sviluppo["frase"], sviluppo[campo].replace("", ASSENTE).values)
        giusti[campo] = pipeline.predict(test["frase"]) == test[campo].replace("", ASSENTE).values
    return giusti


def mcnemar(a, b):
    """Fa il test di McNemar esatto fra due metodi sulle stesse frasi."""
    solo_a, solo_b = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(min(solo_a, solo_b), solo_a + solo_b, 0.5).pvalue if solo_a + solo_b else 1.0
    return solo_a, solo_b, float(p)


def main():
    if not FRASI.exists():
        sys.exit(f"15_interpretazione.py: manca {FRASI.relative_to(ROOT)}, il file con le frasi "
                 "di sviluppo e di test. Senza quel file il confronto non si puo' fare.")
    tutte = pd.read_csv(FRASI).fillna("")
    sviluppo = tutte[tutte["insieme"] == "sviluppo"].reset_index(drop=True)
    test = tutte[tutte["insieme"] == "test"].reset_index(drop=True)
    print(f"Frasi: {len(sviluppo)} di sviluppo, {len(test)} di test\n")
    print("Le regole sono state corrette sulle frasi di sviluppo; i classificatori")
    print("sono addestrati sulle stesse. Il confronto avviene sulle frasi di test,")
    print("mai viste da nessuno dei due metodi.\n")

    risultati = []
    riga_sviluppo, _ = accuratezza_regole(sviluppo)
    giusti_regole = {}
    riga_test, previsioni_test = accuratezza_regole(test, giusti_regole)
    risultati.append({"metodo": "Regole esplicite (nostro sistema)", **riga_test})
    giusti = {"Regole esplicite (nostro sistema)": giusti_regole}

    for nome, modello in classificatori().items():
        riga = {"metodo": nome}
        for campo in CAMPI:
            y_addestramento = sviluppo[campo].replace("", ASSENTE).values
            y_test = test[campo].replace("", ASSENTE).values
            pipeline = make_pipeline(
                TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), modello)
            pipeline.fit(sviluppo["frase"], y_addestramento)
            previsioni = pipeline.predict(test["frase"])
            riga[campo] = round(float((previsioni == y_test).mean()), 3)
        risultati.append(riga)
        giusti[nome] = giusti_modello(modello, sviluppo, test)

    # riferimento banale: la risposta piu' comune nelle frasi di sviluppo, senza
    # leggere la frase
    banale = {"metodo": "Riferimento: sempre la risposta piu' comune"}
    for campo in CAMPI:
        moda = sviluppo[campo].replace("", ASSENTE).value_counts().idxmax()
        banale[campo] = round(float((test[campo].replace("", ASSENTE) == moda).mean()), 3)
    risultati.append(banale)

    tabella = pd.DataFrame(risultati).set_index("metodo")
    tabella["media"] = tabella[CAMPI].mean(axis=1).round(3)

    print("=" * 80)
    print("ACCURATEZZA SULLE 36 FRASI DI TEST")
    print("quota di frasi in cui il campo viene letto correttamente, compresi")
    print("i casi in cui la risposta giusta e' 'informazione assente'")
    print("=" * 80)
    print(tabella.to_string())
    tabella.to_csv(OUT_TAB / "interpretazione_testo.csv")

    print(f"\nPer riferimento, le regole sulle frasi di sviluppo: "
          f"{pd.Series(riga_sviluppo).mean():.3f} di media "
          f"(e' l'insieme su cui sono state corrette, quindi il valore e' ottimistico)")

    print("\n" + "-" * 80)
    print("ESTRAZIONE DI GIORNI E MINUTI sulle frasi di test")
    print("(sono numeri: si estraggono con espressioni regolari, non si classificano)")
    print("-" * 80)
    for campo, chiave in [("giorni", "giorni_settimana"), ("minuti", "minuti_per_seduta")]:
        indicati = test[campo] != ""
        corretti = sum(1 for i, p in enumerate(previsioni_test)
                       if indicati.iloc[i] and p.get(chiave) == int(test[campo].iloc[i]))
        falsi = sum(1 for i, p in enumerate(previsioni_test)
                    if not indicati.iloc[i] and chiave in p)
        print(f"  {campo}: {corretti}/{indicati.sum()} corretti, "
              f"{falsi} estratti per errore da frasi che non li contenevano")

    print("\n" + "-" * 80)
    print("ERRORI DELLE REGOLE SULLE FRASI DI TEST")
    print("-" * 80)
    for i, frase in enumerate(test["frase"]):
        errori = [f"{campo}: atteso {test[campo].iloc[i] or ASSENTE}, "
                  f"letto {previsioni_test[i].get(campo, ASSENTE)}"
                  for campo in CAMPI
                  if (test[campo].iloc[i] or ASSENTE) != previsioni_test[i].get(campo, ASSENTE)]
        if errori:
            print(f'  "{frase[:70]}"')
            for errore in errori:
                print(f"      {errore}")

    # --- parole che scattano solo sul test ---
    vocabolari = {"obiettivo": richiesta.PAROLE_OBIETTIVO, "livello": richiesta.PAROLE_LIVELLO,
                  "attrezzatura": richiesta.PAROLE_ATTREZZATURA}
    def scatta(parola, frasi):
        return any(richiesta._trovata(f.lower(), parola) for f in frasi)
    solo_test = [(campo, valore, parola)
                 for campo, voc in vocabolari.items() for valore, parole in voc.items()
                 for parola in parole
                 if scatta(parola, test["frase"]) and not scatta(parola, sviluppo["frase"])]
    mai = sum(1 for voc in vocabolari.values() for parole in voc.values() for parola in parole
              if not scatta(parola, tutte["frase"]))
    totale = sum(len(parole) for voc in vocabolari.values() for parole in voc.values())
    pd.DataFrame(solo_test, columns=["campo", "valore", "parola"]).to_csv(
        OUT_TAB / "interpretazione_parole_solo_test.csv", index=False)
    originali = {campo: {v: list(p) for v, p in voc.items()} for campo, voc in vocabolari.items()}
    for campo, valore, parola in solo_test:
        vocabolari[campo][valore].remove(parola)
    giusti_senza = {}
    riga_senza, _ = accuratezza_regole(test, giusti_senza)
    for campo, voc in vocabolari.items():
        for valore in voc:
            voc[valore][:] = originali[campo][valore]
    confronto = pd.DataFrame([
        {"regole": "come sono", **riga_test},
        {"regole": "senza le parole che scattano solo sul test", **riga_senza},
        {"regole": "sulle frasi di sviluppo (ottimistico)", **riga_sviluppo},
    ]).set_index("regole")
    confronto["media"] = confronto[CAMPI].mean(axis=1).round(3)
    confronto["parole_tolte"] = [0, len(solo_test), 0]
    confronto.to_csv(OUT_TAB / "interpretazione_senza_parole_test.csv")
    print("\n" + "-" * 80)
    print(f"PAROLE CHE SCATTANO SOLO SUL TEST: {len(solo_test)} su {totale} "
          f"(e {mai} che non scattano su nessuna delle {len(tutte)} frasi)")
    print("-" * 80)
    print(confronto.to_string())

    # --- confronti appaiati e intervalli ---
    righe_conf = []
    for etichetta, g in [("regole come sono", giusti_regole),
                         ("regole senza le parole solo-test", giusti_senza)]:
        for nome in ["Logistic Regression", "Random Forest", "SVM (kernel lineare)"]:
            m = giusti[nome]
            riga = {"regole": etichetta, "contro": nome}
            for campo in CAMPI:
                sa, sb, p = mcnemar(g[campo], m[campo])
                riga[f"{campo}_solo_regole"] = sa
                riga[f"{campo}_solo_modello"] = sb
                riga[f"{campo}_p"] = round(p, 3)
            per_frase = (np.mean([g[c] for c in CAMPI], axis=0)
                         - np.mean([m[c] for c in CAMPI], axis=0))
            media, lo, hi = bootstrap_ic(per_frase)
            riga["differenza_media"] = round(media, 3)
            riga["ic_differenza_media"] = f"[{lo:+.3f}, {hi:+.3f}]"
            righe_conf.append(riga)
        for campo in CAMPI:
            lo, hi = intervallo_wilson(int(g[campo].sum()), len(g[campo]))
            righe_conf.append({"regole": etichetta, "contro": f"Wilson {campo}",
                               "differenza_media": round(float(g[campo].mean()), 3),
                               "ic_differenza_media": f"[{lo:.2f}, {hi:.2f}]"})
    conf = pd.DataFrame(righe_conf)
    conf.to_csv(OUT_TAB / "interpretazione_confronti.csv", index=False)
    print("\n" + "-" * 80)
    print("CONFRONTI APPAIATI (McNemar esatto per campo; bootstrap sulle frasi per la media)")
    print("-" * 80)
    print(conf.to_string(index=False))

    # --- varianti: Random Forest con altri semi, SVM senza pesi bilanciati ---
    medie_rf = []
    for seme in range(20):
        g = giusti_modello(RandomForestClassifier(n_estimators=300, random_state=seme,
                                                  class_weight="balanced"), sviluppo, test)
        medie_rf.append(np.mean([g[c].mean() for c in CAMPI]))
    g_svm = giusti_modello(SVC(kernel="linear", random_state=42), sviluppo, test)
    varianti = pd.DataFrame([
        {"variante": "Random Forest, 20 semi: minimo", "media": round(min(medie_rf), 3)},
        {"variante": "Random Forest, 20 semi: media", "media": round(float(np.mean(medie_rf)), 3)},
        {"variante": "Random Forest, 20 semi: massimo", "media": round(max(medie_rf), 3)},
        {"variante": "SVM senza class_weight bilanciato",
         "media": round(float(np.mean([g_svm[c].mean() for c in CAMPI])), 3),
         "obiettivo": round(float(g_svm["obiettivo"].mean()), 3)},
    ])
    varianti.to_csv(OUT_TAB / "interpretazione_varianti.csv", index=False)
    print("\n" + varianti.to_string(index=False))

    print("\nSalvato: outputs/tables/interpretazione_testo.csv, "
          "interpretazione_parole_solo_test.csv, interpretazione_senza_parole_test.csv, "
          "interpretazione_confronti.csv, interpretazione_varianti.csv")


if __name__ == "__main__":
    main()

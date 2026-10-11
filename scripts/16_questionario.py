"""16 - Prova con risposte reali (questionario).

Confronta la scelta di ogni rispondente con quella del simulatore e prova le regole
di src/richiesta.py su frasi vere. Le scelte sono sbilanciate, quindi il riferimento
non e' il 33%. Prima va salvato l'export in data/valutazione/risposte_questionario.csv.
"""
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation import intervallo_wilson
from src.data_loading import OUT_TAB
from src.matching import ProfiloUtente
from src.richiesta import interpreta, zone_indicate
from src.recommenders import NON_FATTIBILE
from src.scoring import punteggio
from src.simulator import UtenteSimulato, utilita
from src.quality import RIFERIMENTO_EQUILIBRIO
from src.scoring import FATTORE_ENFASI
from scipy.stats import binomtest

RISPOSTE = ROOT / "data" / "valutazione" / "risposte_questionario.csv"
FRASI = ROOT / "data" / "valutazione" / "frasi_utente.csv"
PROCESSED = ROOT / "data" / "processed"
ASSENTE = "(assente)"

# i tre programmi del questionario: non vanno cambiati
PROGRAMMI_PROPOSTI = {
    "A": "Malone Beginner Strength",
    "B": "Warlock",
    "C": "Aesthetic Hypertrophy for the Natural Lifter",
}

# Google Form usa il testo della domanda come intestazione: per ogni colonna cerco
# una parola. Se cambiano le domande, va adattato.
COLONNE = {
    "frase": "parole tue",
    "obiettivo": "obiettivo principale",
    "livello": "che punto sei",
    "attrezzatura": "Dove e con cosa",
    "giorni": "giorni a settimana",
    "minuti": "minuti hai",
    "zone": "zona su cui",
    "scelta": "Quale sceglieresti",
}

DA_ETICHETTA_OBIETTIVO = {
    "Tonificare e definire": "Muscle & Sculpting",
    "Aumentare la massa muscolare": "Bodybuilding",
    "Forza e massa insieme": "Powerbuilding",
    "Diventare piu' forte sulle alzate": "Powerlifting",
    "Diventare più forte sulle alzate": "Powerlifting",
    "Prestazione sportiva": "Athletics",
    "Allenarmi a corpo libero": "Bodyweight Fitness",
    "Pesistica olimpica": "Olympic Weightlifting",
}
DA_ETICHETTA_LIVELLO = {
    "Sto iniziando adesso": "Beginner",
    "Ho qualche mese di esperienza": "Novice",
    "Mi alleno con continuita' da tempo": "Intermediate",
    "Mi alleno con continuità da tempo": "Intermediate",
    "Sono a un livello avanzato": "Advanced",
}
DA_ETICHETTA_ATTREZZATURA = {
    "Solo un paio di manubri": "Dumbbell Only",
    "A casa, poca attrezzatura": "At Home",
    "Home gym con bilanciere": "Garage Gym",
    "Palestra attrezzata": "Full Gym",
}
DA_ETICHETTA_ZONA = {"Gambe e glutei": "gambe", "Schiena": "dorso", "Petto": "petto",
                     "Spalle": "spalle", "Braccia": "braccia", "Addome": "core"}


CASO = 1 / 3  # tre programmi, una scelta


def pvalue_binomiale(successi, prove, p0=CASO):
    """Calcola la probabilita' di avere almeno questi successi rispondendo a caso.

    Binomiale esatto: con poche osservazioni l'approssimazione normale non regge.
    """
    if not prove:
        return float("nan")
    return sum(math.comb(prove, i) * p0 ** i * (1 - p0) ** (prove - i)
               for i in range(successi, prove + 1))


def trova_colonna(dati, parola):
    for colonna in dati.columns:
        if parola.lower() in colonna.lower():
            return colonna
    raise KeyError(f"colonna non trovata per '{parola}'. Colonne disponibili: "
                   f"{list(dati.columns)}")


def confronto_su_lingua_vera(dati, colonne):
    """Mette alla prova regole e classificatori sulle frasi del questionario.

    I classificatori si addestrano sulle stesse frasi di sviluppo di 15.
    """
    if not FRASI.exists():
        return None
    tutte = pd.read_csv(FRASI).fillna("")
    sviluppo = tutte[tutte["insieme"] == "sviluppo"].reset_index(drop=True)

    campi = {"obiettivo": DA_ETICHETTA_OBIETTIVO,
             "livello": DA_ETICHETTA_LIVELLO,
             "attrezzatura": DA_ETICHETTA_ATTREZZATURA}

    frasi, veri = [], {campo: [] for campo in campi}
    for _, risposta in dati.iterrows():
        frase = str(risposta[colonne["frase"]]).strip()
        etichette = {campo: mappa.get(str(risposta[colonne[campo]]).strip())
                     for campo, mappa in campi.items()}
        if not frase or any(v is None for v in etichette.values()):
            continue
        frasi.append(frase)
        for campo in campi:
            veri[campo].append(etichette[campo])
    if not frasi:
        return None

    modelli = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42,
                                                class_weight="balanced"),
        "SVM (kernel lineare)": SVC(kernel="linear", class_weight="balanced",
                                    random_state=42),
    }

    letture = [interpreta(f) for f in frasi]
    riga_regole, copertura = {"metodo": "Regole esplicite (nostro sistema)"}, {}
    for campo in campi:
        predetto = [lettura.get(campo, ASSENTE) for lettura in letture]
        atteso = veri[campo]
        riga_regole[campo] = round(float(np.mean([p == a for p, a in zip(predetto, atteso)])), 3)
        risposte_date = [(p, a) for p, a in zip(predetto, atteso) if p != ASSENTE]
        copertura[campo] = (len(risposte_date), len(frasi),
                            sum(1 for p, a in risposte_date if p == a))

    righe = [riga_regole]
    concentrazione = {}
    for nome, modello in modelli.items():
        riga = {"metodo": nome}
        for campo in campi:
            y = sviluppo[campo].replace("", ASSENTE).values
            pipeline = make_pipeline(
                TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), modello)
            pipeline.fit(sviluppo["frase"], y)
            predetto = list(pipeline.predict(frasi))
            riga[campo] = round(float(np.mean([p == a for p, a in zip(predetto, veri[campo])])), 3)
            # se un modello dice sempre la stessa cosa non sta leggendo la frase
            piu_frequente = max(set(predetto), key=predetto.count)
            concentrazione[(nome, campo)] = (piu_frequente,
                                             predetto.count(piu_frequente), len(predetto))
        righe.append(riga)

    # Senza il punteggio di chi risponde sempre il valore piu' comune, i numeri
    # sopra non si leggono.
    riga_banale = {"metodo": "Riferimento: sempre la risposta piu' comune"}
    for campo in campi:
        conteggi = {v: veri[campo].count(v) for v in set(veri[campo])}
        riga_banale[campo] = round(max(conteggi.values()) / len(frasi), 3)
    righe.append(riga_banale)

    tabella = pd.DataFrame(righe).set_index("metodo")
    tabella["media"] = tabella[list(campi)].mean(axis=1).round(3)
    tabella.insert(0, "frasi", len(frasi))
    return tabella, copertura, concentrazione


def main():
    if not RISPOSTE.exists():
        sys.exit(f"16_questionario.py: manca {RISPOSTE.relative_to(ROOT)}. Esporta le "
                 "risposte del modulo in CSV e salvalo li'.")

    dati = pd.read_csv(RISPOSTE).fillna("")
    colonne = {chiave: trova_colonna(dati, parola) for chiave, parola in COLONNE.items()}
    catalogo = pd.read_csv(PROCESSED / "catalogo_con_qualita.csv")
    quote = pd.read_csv(PROCESSED / "quote_muscolari.csv", index_col=0)
    print(f"Risposte raccolte: {len(dati)}\n")

    indici_proposti = {
        lettera: catalogo.index[catalogo["title"] == titolo][0]
        for lettera, titolo in PROGRAMMI_PROPOSTI.items()
        if (catalogo["title"] == titolo).any()
    }
    if len(indici_proposti) < 3:
        print("Attenzione: non tutti i programmi proposti sono nel catalogo.")

    concordi_sistema, concordi_simulatore, scelte = 0, 0, 0
    righe_lingua, righe_scelta, esclusi = [], [], 0
    candidati = list(indici_proposti.values())
    lettera_di = {i: l for l, i in indici_proposti.items()}

    def scelta_simulatore(profilo, zone=(), scala_capacita=1.0, rumore=False, seme=0):
        """Sceglie il programma che l'utente "finto" preferisce fra i tre.

        Ha la capacita' stimata dal sistema; con 'zone' le zone dichiarate pesano di piu'.
        """
        distribuzione = {g: RIFERIMENTO_EQUILIBRIO[g] * (FATTORE_ENFASI if g in zone else 1)
                         for g in RIFERIMENTO_EQUILIBRIO}
        totale = sum(distribuzione.values())
        finto = UtenteSimulato(
            obiettivo=profilo.obiettivo, livello=profilo.livello,
            attrezzatura=profilo.attrezzatura, giorni_settimana=profilo.giorni_settimana,
            minuti_per_seduta=profilo.minuti_per_seduta,
            capacita_serie=profilo.capacita_stimata * scala_capacita,
            distribuzione_desiderata={g: v / totale for g, v in distribuzione.items()},
            seme=seme)
        u = utilita(catalogo, quote, finto, con_rumore=rumore)
        return lettera_di[max(candidati, key=lambda i: u.iloc[i])]

    def scelta_sistema(profilo):
        p = punteggio(catalogo, profilo).fillna(NON_FATTIBILE)
        return lettera_di[max(candidati, key=lambda i: p.iloc[i])]

    for numero, (_, risposta) in enumerate(dati.iterrows()):
        obiettivo = DA_ETICHETTA_OBIETTIVO.get(str(risposta[colonne["obiettivo"]]).strip())
        livello = DA_ETICHETTA_LIVELLO.get(str(risposta[colonne["livello"]]).strip())
        attrezzatura = DA_ETICHETTA_ATTREZZATURA.get(str(risposta[colonne["attrezzatura"]]).strip())
        if not (obiettivo and livello and attrezzatura):
            esclusi += 1
            print(f"  risposta esclusa: etichette non riconosciute "
                  f"({risposta[colonne['obiettivo']]!r}, {risposta[colonne['livello']]!r}, "
                  f"{risposta[colonne['attrezzatura']]!r})")
            continue
        giorni = int(str(risposta[colonne["giorni"]]).strip() or 3)
        testo_minuti = str(risposta[colonne["minuti"]]).split()[0]
        minuti = int(testo_minuti) if testo_minuti.isdigit() else 60
        zone = tuple(DA_ETICHETTA_ZONA[z.strip()]
                     for z in str(risposta[colonne["zone"]]).split(",")
                     if z.strip() in DA_ETICHETTA_ZONA)[:2]

        profilo = ProfiloUtente(obiettivo, livello, attrezzatura, giorni, minuti,
                                enfasi=zone)

        # --- 1. la scelta ---
        # Google Form riporta tutto il testo dell'opzione: tengo la lettera.
        testo_scelta = str(risposta[colonne["scelta"]]).strip()
        trovata = re.search(r"Programma\s*\**\s*([ABC])\b", testo_scelta, re.I)
        lettera = trovata.group(1).upper() if trovata else testo_scelta[:1].upper()
        if lettera in indici_proposti:
            scelte += 1
            # Il modulo diceva di immaginare l'attrezzatura per tutti e tre: uso la
            # palestra. Con l'attrezzatura vera molte previsioni sarebbero forzate.
            in_palestra = ProfiloUtente(obiettivo, livello, "Full Gym", giorni, minuti,
                                        enfasi=zone)
            riga = {"scelta": lettera,
                    "sistema": scelta_sistema(in_palestra),
                    "simulatore": scelta_simulatore(in_palestra),
                    "sistema_attrezzatura_dichiarata": scelta_sistema(profilo),
                    "simulatore_attrezzatura_dichiarata": scelta_simulatore(profilo),
                    "simulatore_con_zone": scelta_simulatore(in_palestra, zone=zone)}
            for scala in (0.5, 0.75, 1.25, 1.5, 2.0):
                riga[f"simulatore_capacita_x{scala}"] = scelta_simulatore(in_palestra,
                                                                          scala_capacita=scala)
            # Un seme per persona, come per gli utenti simulati: un seme unico
            # sposterebbe allo stesso modo A, B e C per tutti.
            for ripetizione in range(20):
                riga[f"simulatore_rumore_{ripetizione}"] = scelta_simulatore(
                    in_palestra, rumore=True, seme=ripetizione * 1000 + numero)
            righe_scelta.append(riga)
            concordi_sistema += riga["sistema"] == lettera
            concordi_simulatore += riga["simulatore"] == lettera

        # --- 2. la lingua ---
        frase = str(risposta[colonne["frase"]]).strip()
        if frase:
            letto = interpreta(frase)
            letto_zone = zone_indicate(frase)
            righe_lingua.append({
                "obiettivo": letto.get("obiettivo") == obiettivo,
                "obiettivo_espresso": "obiettivo" in letto,
                "livello": letto.get("livello") == livello,
                "livello_espresso": "livello" in letto,
                "attrezzatura": letto.get("attrezzatura") == attrezzatura,
                "attrezzatura_espressa": "attrezzatura" in letto,
                "giorni": letto.get("giorni_settimana") == giorni,
                "giorni_espresso": "giorni_settimana" in letto,
                "minuti": letto.get("minuti_per_seduta") == minuti,
                "minuti_espresso": "minuti_per_seduta" in letto,
                "zone": bool(set(letto_zone) & set(zone)) if zone else False,
                "zone_espresse": bool(letto_zone),
            })

    print("=" * 74)
    print("1. IL SIMULATORE SOMIGLIA ALLE PERSONE VERE?")
    print("=" * 74)
    if scelte:
        print(f"  risposte utilizzabili ................. {scelte}")
        print(f"  scelta prevista dal simulatore ....... {concordi_simulatore}/{scelte} "
              f"({concordi_simulatore/scelte*100:.0f}%)")
        print(f"  scelta prevista dal sistema .......... {concordi_sistema}/{scelte} "
              f"({concordi_sistema/scelte*100:.0f}%)")
        print("  (attrezzatura: palestra per tutti, come chiedeva il modulo)")
        for etichetta, concordi in [("simulatore", concordi_simulatore),
                                    ("sistema", concordi_sistema)]:
            lo, hi = intervallo_wilson(concordi, scelte)
            print(f"  {etichetta:<12} IC 95% [{lo*100:.0f}%; {hi*100:.0f}%] | "
                  f"p contro un terzo = {pvalue_binomiale(concordi, scelte):.4f}")
        print("  ATTENZIONE: un terzo NON e' il riferimento giusto, perche' le scelte reali\n"
              "  sono sbilanciate. I riferimenti corretti sono nella sezione 1b.")
    else:
        print("  nessuna risposta utilizzabile")

    print("\n" + "=" * 74)
    print("2. LE REGOLE FUNZIONANO SULLA LINGUA VERA?")
    print("=" * 74)
    if righe_lingua:
        L = pd.DataFrame(righe_lingua)
        for campo in ["obiettivo", "livello", "attrezzatura"]:
            espresso = L[f"{campo}_espress" + ("o" if campo != "attrezzatura" else "a")]
            print(f"  {campo:<14} riconosciuto in {espresso.mean()*100:.0f}% delle frasi, "
                  f"corretto nel {L.loc[espresso, campo].mean()*100:.0f}% di quelle")
        # Giorni, minuti e zone contano solo se la frase li nomina.
        for campo in ["giorni", "minuti", "zone"]:
            espresso = L[f"{campo}_espress" + ("e" if campo == "zone" else "o")]
            if espresso.any():
                print(f"  {campo:<14} riconosciuto in {espresso.mean()*100:.0f}% delle frasi, "
                      f"corretto nel {L.loc[espresso, campo].mean()*100:.0f}% di quelle")
            else:
                print(f"  {campo:<14} non nominato in nessuna frase")
        righe = []
        for campo, suffisso in [("obiettivo", "o"), ("livello", "o"), ("attrezzatura", "a"),
                                ("giorni", "o"), ("minuti", "o"), ("zone", "e")]:
            espresso = L[f"{campo}_espress{suffisso}"]
            giusti = int(L.loc[espresso, campo].sum())
            righe.append({"campo": campo, "frasi": len(L),
                          "riconosciuto_dalle_regole": int(espresso.sum()),
                          "corretto_fra_quelle": giusti,
                          "quota_riconosciuto": round(float(espresso.mean()), 3),
                          "quota_corretto": round(giusti / espresso.sum(), 3) if espresso.any() else None})
        pd.DataFrame(righe).to_csv(OUT_TAB / "questionario_lingua.csv", index=False)

    # --- 3. regole contro classificatori, sulle frasi vere ---
    # Il bersaglio e' cio' che la persona ha selezionato nel modulo. Chi non risponde
    # sbaglia: e' l'unico modo di confrontare un metodo che puo' tacere con tre che
    # rispondono sempre.
    confronto = confronto_su_lingua_vera(dati, colonne)
    if confronto is not None:
        tabella, copertura, concentrazione = confronto
        print("\n" + "=" * 74)
        print("3. REGOLE CONTRO CLASSIFICATORI, SULLE FRASI VERE")
        print("=" * 74)
        print("  bersaglio: il campo che la persona ha selezionato nel modulo")
        print("  (chi non risponde conta come errore, per entrambi i metodi)\n")
        print(tabella.to_string())
        print("\n  Le regole rispondono solo quando riconoscono qualcosa:")
        for campo, (quante, su, giuste) in copertura.items():
            if quante:
                print(f"    {campo:<14} rispondono {quante}/{su} volte, "
                      f"giuste {giuste}/{quante}")
            else:
                print(f"    {campo:<14} non rispondono mai ({su} frasi)")
        print("\n  Cosa predicono davvero i classificatori (previsione piu' frequente):")
        for (nome, campo), (valore, quante, su) in concentrazione.items():
            if quante / su >= 0.7:
                print(f"    {nome:<22} {campo:<14} dice '{valore}' {quante}/{su} volte")
        tabella.to_csv(OUT_TAB / "interpretazione_lingua_vera.csv")

    # --- 1b. riferimenti per l'accordo ---
    S = pd.DataFrame(righe_scelta)
    vere = S["scelta"]
    piu_scelta = vere.value_counts().idxmax()
    righe_rif = []
    for nome, col in [("simulatore", "simulatore"), ("sistema", "sistema"),
                      ("simulatore, attrezzatura dichiarata", "simulatore_attrezzatura_dichiarata"),
                      ("sistema, attrezzatura dichiarata", "sistema_attrezzatura_dichiarata"),
                      ("simulatore con le zone dichiarate", "simulatore_con_zone")]:
        giusti = S[col] == vere
        k, n = int(giusti.sum()), len(S)
        lo, hi = intervallo_wilson(k, n)
        # accordo atteso per caso con le stesse proporzioni, come nel kappa di Cohen
        p0 = float(sum(vere.value_counts(normalize=True).get(l, 0) *
                       S[col].value_counts(normalize=True).get(l, 0) for l in "ABC"))
        kappa = (k / n - p0) / (1 - p0) if p0 < 1 else float("nan")
        sempre = vere == piu_scelta
        solo_pred, solo_sempre = int((giusti & ~sempre).sum()), int((~giusti & sempre).sum())
        p_mc = (binomtest(min(solo_pred, solo_sempre), solo_pred + solo_sempre, 0.5).pvalue
                if solo_pred + solo_sempre else 1.0)
        righe_rif.append({
            "previsione": nome, "giuste": k, "su": n, "ic": f"[{lo*100:.0f}%, {hi*100:.0f}%]",
            "p_contro_un_terzo": round(pvalue_binomiale(k, n), 4),
            "accordo_atteso_per_caso": round(p0, 3),
            "p_contro_caso_stesse_proporzioni": round(float(binomtest(k, n, p0, alternative="greater").pvalue), 3),
            "kappa": round(kappa, 3),
            "solo_previsione_giusta": solo_pred, "solo_sempre_la_piu_scelta_giusta": solo_sempre,
            "p_mcnemar_contro_sempre_la_piu_scelta": round(float(p_mc), 3)})
    righe_rif.append({"previsione": f"sempre '{piu_scelta}' (la piu' scelta)",
                      "giuste": int((vere == piu_scelta).sum()), "su": len(S)})
    rif = pd.DataFrame(righe_rif)
    rif.to_csv(OUT_TAB / "questionario_scelte.csv", index=False)
    accordo = int((S["simulatore"] == S["sistema"]).sum())
    sens = [{"variante": "simulatore e sistema scelgono lo stesso programma",
             "valore": accordo, "su": len(S)}]
    for scala in (0.5, 0.75, 1.25, 1.5, 2.0):
        sens.append({"variante": f"capacita' x{scala}",
                     "valore": int((S[f"simulatore_capacita_x{scala}"] == vere).sum()), "su": len(S)})
    rumore = [int((S[f"simulatore_rumore_{s}"] == vere).sum()) for s in range(20)]
    sens += [{"variante": "con il rumore, 20 semi: minimo", "valore": min(rumore), "su": len(S)},
             {"variante": "con il rumore, 20 semi: media", "valore": round(float(np.mean(rumore)), 1), "su": len(S)},
             {"variante": "con il rumore, 20 semi: massimo", "valore": max(rumore), "su": len(S)}]
    pd.DataFrame(sens).to_csv(OUT_TAB / "questionario_sensibilita.csv", index=False)
    print("\n" + "=" * 74)
    print("1b. CONTRO QUALI RIFERIMENTI (scelte reali: " +
          ", ".join(f"{l} {int((vere == l).sum())}" for l in "ABC") + ")")
    print("=" * 74)
    print(rif.to_string(index=False))
    print(f"\n  simulatore e sistema scelgono lo stesso programma in {accordo}/{len(S)} casi")
    print(f"  risposte escluse: {esclusi}")
    print(pd.DataFrame(sens).to_string(index=False))

    pd.DataFrame([{
        "risposte": len(dati), "scelte_valutate": scelte,
        "accordo_simulatore_%": round(concordi_simulatore / scelte * 100, 1) if scelte else None,
        "accordo_sistema_%": round(concordi_sistema / scelte * 100, 1) if scelte else None,
        "ic_simulatore": "[{:.0f}%, {:.0f}%]".format(*[x * 100 for x in intervallo_wilson(concordi_simulatore, scelte)]) if scelte else None,
        "ic_sistema": "[{:.0f}%, {:.0f}%]".format(*[x * 100 for x in intervallo_wilson(concordi_sistema, scelte)]) if scelte else None,
        "p_simulatore": round(pvalue_binomiale(concordi_simulatore, scelte), 4) if scelte else None,
        "p_sistema": round(pvalue_binomiale(concordi_sistema, scelte), 4) if scelte else None,
    }]).to_csv(OUT_TAB / "questionario_convalida.csv", index=False)
    print("\nSalvati: outputs/tables/questionario_convalida.csv, questionario_lingua.csv, "
          "questionario_scelte.csv, questionario_sensibilita.csv")


if __name__ == "__main__":
    main()

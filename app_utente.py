"""Demo per l'utente finale: una pagina per la richiesta, una per i risultati.

Niente punteggi ne' metriche, solo frasi.
Avvio:  streamlit run app_utente.py
"""
import sys
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.catalog import ATTREZZATURE, LIVELLI, OBIETTIVI
from src.matching import ProfiloUtente
from src.scoring import punteggio, rilevanza, componenti
from src.adaptive import StimaAdattiva, PESO_CORREZIONE
from src.scoring import voto_sostenibilita
from src.richiesta import (interpreta, zone_indicate, motivo_astensione, ETICHETTE,
                            GIORNI_MIN, GIORNI_MAX, MINUTI_MIN, MINUTI_MAX,
                            VALORI_INIZIALI)
from src.scheda import schede_disponibili, carica_scheda, giorni_della_settimana, rpe_medio
from src.spiegazioni import (punti_di_forza, punti_di_attenzione, punto_enfasi,
                             titolo_da_mostrare, ZONE_IN_ITALIANO)
from src.evaluation import ordina
from src.stile import CSS, pillola, punto, riga_esercizio

st.set_page_config(page_title="Trova il tuo allenamento", page_icon="🏋", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

ETICHETTE_OBIETTIVO = {
    "Muscle & Sculpting": "Tonificare e definire",
    "Bodybuilding": "Aumentare la massa muscolare",
    "Powerbuilding": "Forza e massa insieme",
    "Powerlifting": "Diventare piu' forte sulle alzate",
    "Athletics": "Prestazione sportiva",
    "Bodyweight Fitness": "Allenarmi a corpo libero",
    "Olympic Weightlifting": "Pesistica olimpica",
}
ETICHETTE_LIVELLO = {
    "Beginner": "Sto iniziando adesso",
    "Novice": "Ho qualche mese di esperienza",
    "Intermediate": "Mi alleno con continuita' da tempo",
    "Advanced": "Sono a un livello avanzato",
}
ETICHETTE_ATTREZZATURA = {
    "Dumbbell Only": "Solo un paio di manubri",
    "At Home": "A casa, poca attrezzatura",
    "Garage Gym": "Home gym con bilanciere",
    "Full Gym": "Palestra attrezzata",
}
DESCRIZIONE_LIVELLO = {1: "chi inizia adesso", 2: "chi ha qualche mese di esperienza",
                       3: "chi si allena con continuita'", 4: "chi e' a un livello avanzato"}
ETICHETTE_ORDINE = ["IL PIU' ADATTO", "SECONDA SCELTA", "TERZA SCELTA"]
# L'attrezzatura non ha un valore iniziale: e' l'unico vincolo rigido, e qualunque
# valore sbaglierebbe per qualcuno. Meglio chiederla. Gli altri valori iniziali
# sono in src/richiesta.py.


def etichetta_ordine(posizione):
    if posizione < len(ETICHETTE_ORDINE):
        return ETICHETTE_ORDINE[posizione]
    return f"{posizione + 1}a POSIZIONE"


@st.cache_data
def carica_catalogo():
    return pd.read_csv(ROOT / "data" / "processed" / "catalogo_con_qualita.csv")


@st.cache_data
def scheda_di(titolo):
    return carica_scheda(titolo)


catalogo = carica_catalogo()
st.session_state.setdefault("profilo", None)
st.session_state.setdefault("testo_richiesta", "")
st.session_state.setdefault("riconosciuti", {})
st.session_state.setdefault("scartati", [])
# l'ultimo scarto, per poterlo raccontare e annullare
st.session_state.setdefault("effetto", None)
st.session_state.setdefault("aperto", None)
st.session_state.setdefault("quanti", 3)


# ==========================================================================
# PAGINA 1 - la richiesta
# ==========================================================================
if st.session_state.profilo is None:
    # larga quasi quanto lo schermo e un po' piu' in basso, per non stare attaccata
    # al bordo
    st.markdown(
        '<style>'
        '.block-container { max-width: min(1700px, 92vw) !important;'
        ' width: min(1700px, 92vw) !important;'
        ' margin-left: auto !important; margin-right: auto !important;'
        ' padding-top: clamp(2rem, 9vh, 8rem) !important;'
        ' padding-left: 0 !important; padding-right: 0 !important; }'
        '</style>', unsafe_allow_html=True)
    numero_programmi = f"{len(catalogo):,}".replace(",", ".")
    st.markdown(
        '<div class="eroe">'
        f'<div class="occhiello">{numero_programmi} programmi reali</div>'
        '<h1>Trova il <em>tuo</em> allenamento</h1>'
        '<p>Raccontami come ti alleni e cosa vuoi ottenere. Ti propongo i programmi '
        'piu\' adatti a te, con la scheda gia\' pronta da seguire.</p></div>',
        unsafe_allow_html=True)

    with st.form("richiesta"):
        sinistra, destra = st.columns([1.25, 1], gap="large")
        with sinistra:
            testo = st.text_area(
                "Cosa stai cercando?",
                placeholder="Vorrei tonificare, ho solo dei manubri a casa e tre "
                            "giorni a settimana da 40 minuti",
                height=150)
            st.markdown('<p class="nota">Scrivilo a parole tue, oppure usa i campi '
                        'accanto.</p>', unsafe_allow_html=True)
        with destra:
            alto1, alto2 = st.columns(2, gap="medium")
            with alto1:
                obiettivo = st.selectbox(
                    "Obiettivo", OBIETTIVI,
                    index=OBIETTIVI.index(VALORI_INIZIALI["obiettivo"]),
                    format_func=lambda v: ETICHETTE_OBIETTIVO[v])
                attrezzatura = st.selectbox("Dove ti alleni", list(ATTREZZATURE), index=None,
                                            placeholder="Scegli…",
                                            format_func=lambda v: ETICHETTE_ATTREZZATURA[v])
            with alto2:
                livello = st.selectbox(
                    "A che punto sei", list(LIVELLI),
                    index=list(LIVELLI).index(VALORI_INIZIALI["livello"]),
                    format_func=lambda v: ETICHETTE_LIVELLO[v])
                giorni = st.slider("Giorni a settimana", GIORNI_MIN, GIORNI_MAX,
                                   VALORI_INIZIALI["giorni_settimana"])
            minuti = st.slider("Minuti per allenamento", MINUTI_MIN, MINUTI_MAX,
                               VALORI_INIZIALI["minuti_per_seduta"], step=5)
        zone = st.pills(
            "C'e' una zona su cui vuoi concentrarti? (facoltativo, massimo due)",
            list(ZONE_IN_ITALIANO), selection_mode="multi",
            format_func=lambda z: ZONE_IN_ITALIANO[z].capitalize())
        inviato = st.form_submit_button("Trova i programmi adatti a me", type="primary",
                                        use_container_width=True)

    if inviato and len(zone) > 2:
        st.error("Puoi scegliere al massimo due zone: togline una e riprova.")
        st.stop()
    if inviato:
        campi = {"obiettivo": obiettivo, "livello": livello, "attrezzatura": attrezzatura,
                 "giorni_settimana": giorni, "minuti_per_seduta": minuti,
                 "enfasi": tuple(zone)}
        riconosciuti = interpreta(testo) if testo.strip() else {}
        zone_dal_testo = zone_indicate(testo) if testo.strip() else ()
        if zone_dal_testo:
            riconosciuti["enfasi"] = zone_dal_testo
        # Conta come scelto solo un campo spostato dal valore iniziale: chi non l'ha
        # toccato non va avvisato di un contrasto col testo.
        scelti = {campo for campo, valore in campi.items()
                  if valore is not None and valore != VALORI_INIZIALI.get(campo)}

        if "attrezzatura" not in riconosciuti and attrezzatura is None:
            motivo = motivo_astensione(testo) if testo.strip() else None
            if motivo == "conflitto":
                st.error("Hai nominato la palestra e anche un'altra attrezzatura, e non "
                         "so quale considerare. Scegli nel campo «Dove ti alleni» quella "
                         "che hai davvero a disposizione quando ti alleni.")
            elif motivo == "dubbio":
                st.error("Hai nominato la palestra, ma non ho capito se ci vai. Dimmi dove "
                         "ti alleni nel campo «Dove ti alleni»: è l'unica cosa che non posso "
                         "indovinare, perché da lì dipende quali programmi puoi davvero fare.")
            else:
                st.error("Non ho capito dove ti alleni. Sceglilo nel campo «Dove ti "
                         "alleni»: è l'unica cosa che non posso indovinare, perché da lì "
                         "dipende quali programmi puoi davvero fare.")
            st.stop()

        # Se testo e campo scelto non coincidono vince il testo, l'ultima cosa scritta
        # dalla persona. Lo si dice, per non cambiare le scelte di nascosto.
        sovrascritti = {campo: (campi[campo], valore) for campo, valore in riconosciuti.items()
                        if campo in scelti and campi[campo] != valore}
        # Anche i campi rimasti al valore iniziale si dicono.
        iniziali = [campo for campo in VALORI_INIZIALI
                    if campo != "enfasi" and campo not in riconosciuti and campo not in scelti]
        campi.update(riconosciuti)
        st.session_state.update(profilo=campi, riconosciuti=riconosciuti,
                                sovrascritti=sovrascritti, iniziali=iniziali,
                                attrezzatura_dal_testo="attrezzatura" in riconosciuti,
                                testo_richiesta=testo.strip(), scartati=[], effetto=None,
                                aperto=None, quanti=3)
        st.rerun()
    st.stop()


# ==========================================================================
# PAGINA 3 - la scheda del programma scelto
# ==========================================================================
def mostra_scheda(titolo):
    riga = catalogo[catalogo["title"] == titolo].iloc[0]

    if st.button("← Torna ai consigli"):
        st.session_state.aperto = None
        st.rerun()

    if not schede_disponibili():
        st.error("Le schede non sono disponibili: esegui prima "
                 "`python scripts/14_scheda.py`.")
        st.stop()

    scheda = scheda_di(titolo)
    settimane = sorted(int(w) for w in scheda["week"].dropna().unique())

    testata, selettore = st.columns([2.6, 1], gap="medium")
    with testata:
        pillole_pagina = "".join([
            pillola(f'{riga["settimane"]:.0f}', "settimane"),
            pillola(f'{riga["giorni_settimana"]:.0f}', "a settimana"),
            pillola(f'{riga["time_per_workout"]:.0f}\'', "a seduta"),
            pillola(f'{riga["volume_settimanale"]:.0f}', "serie a sett."),
        ])
        st.markdown(f'<div class="eroe"><div class="occhiello">La scheda completa</div>'
                    f'<h1>{escape(titolo_da_mostrare(titolo))}</h1>'
                    f'<div class="pillole" style="margin-top:18px">{pillole_pagina}</div>'
                    f'</div>', unsafe_allow_html=True)
    with selettore:
        settimana = st.selectbox("Settimana", settimane,
                                 format_func=lambda w: f"Settimana {w} di {len(settimane)}")
        settimana_scelta = scheda[scheda["week"] == settimana]
        pillole_settimana = "".join([
            pillola(f'{settimana_scelta["sets"].sum():.0f}', "serie"),
            pillola((lambda r: f"{r:.1f}" if r == r else "n.d.")(rpe_medio(settimana_scelta)),
                    "RPE medio"),
            pillola(f'{settimana_scelta[["week", "day"]].drop_duplicates().shape[0]}',
                    "allenamenti"),
        ])
        st.markdown(f'<div class="pillole" style="margin-top:12px">{pillole_settimana}</div>',
                    unsafe_allow_html=True)

    giorni = giorni_della_settimana(scheda, settimana)
    if not giorni:
        st.info("Nessun esercizio registrato per questa settimana.")
        return

    elenco = list(giorni.items())
    per_riga = 4 if len(elenco) >= 4 else max(len(elenco), 1)
    for inizio_riga in range(0, len(elenco), per_riga):
        blocco = elenco[inizio_riga:inizio_riga + per_riga]
        colonne_giorni = st.columns(per_riga, gap="medium")
        for (giorno, tabella), colonna_giorno in zip(blocco, colonne_giorni):
            righe_html = [riga_esercizio(numero, esercizio["Esercizio"], esercizio["Serie"])
                          for numero, (_, esercizio) in enumerate(tabella.iterrows(), start=1)]
            with colonna_giorno:
                st.markdown(
                    f'<div class="giorno"><div class="intestazione-giorno">'
                    f'<h4>Allenamento {giorno}</h4>'
                    f'<span class="conteggio">{len(tabella)} esercizi</span></div>'
                    f'{"".join(righe_html)}</div>', unsafe_allow_html=True)

    st.markdown('<p class="nota">RPE indica quanto deve essere impegnativa la serie, da 1 '
                'a 10: 8 significa che potresti fare ancora circa due ripetizioni.</p>',
                unsafe_allow_html=True)


if st.session_state.aperto is not None:
    mostra_scheda(st.session_state.aperto)
    st.stop()

# ==========================================================================
# PAGINA 2 - i risultati
# ==========================================================================
campi = st.session_state.profilo
profilo = ProfiloUtente(campi["obiettivo"], campi["livello"], campi["attrezzatura"],
                        campi["giorni_settimana"], campi["minuti_per_seduta"],
                        enfasi=tuple(campi.get("enfasi", ())))

if st.button("← Modifica la richiesta"):
    st.session_state.update(profilo=None, aperto=None)
    st.rerun()

# --- riepilogo della richiesta ---
riconosciuti = st.session_state.riconosciuti
pillole_profilo = "".join([
    pillola(ETICHETTE_OBIETTIVO[campi["obiettivo"]].split(" e ")[0], "obiettivo"),
    pillola(campi["livello"].replace("Beginner", "Principiante")
            .replace("Novice", "Base").replace("Intermediate", "Intermedio")
            .replace("Advanced", "Avanzato"), "livello"),
    pillola(ETICHETTE_ATTREZZATURA[campi["attrezzatura"]].split(",")[0],
            "attrezzatura, " + ("dal testo" if st.session_state.get("attrezzatura_dal_testo")
                                else "scelta da te")),
    pillola(f'{campi["giorni_settimana"]}', "giorni a sett."),
    pillola(f'{campi["minuti_per_seduta"]}\'', "per seduta"),
] + ([pillola(" e ".join(ZONE_IN_ITALIANO[z] for z in campi["enfasi"]), "focus")]
     if campi.get("enfasi") else []))
citazione = ""
if st.session_state.testo_richiesta:
    citazione = (f'<p class="citazione">&ldquo;{escape(st.session_state.testo_richiesta)}'
                 f'&rdquo;</p>')
def in_italiano(campo, valore):
    if campo == "enfasi":
        return " e ".join(ZONE_IN_ITALIANO[z] for z in valore) or "nessuna"
    traduzioni = {"obiettivo": ETICHETTE_OBIETTIVO, "livello": ETICHETTE_LIVELLO,
                  "attrezzatura": ETICHETTE_ATTREZZATURA}
    return traduzioni.get(campo, {}).get(valore, valore)


nota_lettura = ""
if riconosciuti:
    nota_lettura = ('<p class="nota" style="margin-top:10px">Ho ricavato da quello che '
                    'hai scritto: ' +
                    ", ".join(f"{ETICHETTE[k]} ({in_italiano(k, v)})"
                              for k, v in riconosciuti.items()) + '.</p>')
iniziali = st.session_state.get("iniziali", [])
if iniziali:
    nota_lettura += ('<p class="nota" style="margin-top:6px">Non li hai indicati, e ho '
                     'usato i valori di partenza del modulo: ' +
                     ", ".join(f"{ETICHETTE[k]} {in_italiano(k, campi[k])}" for k in iniziali) +
                     '. Se non ti corrispondono, torna indietro e cambiali.</p>')
sovrascritti = st.session_state.get("sovrascritti", {})
if sovrascritti:
    elenco = "; ".join(
        f'{ETICHETTE[campo]}: ho usato <b>{in_italiano(campo, nuovo_valore)}</b> '
        f'invece di {in_italiano(campo, vecchio_valore)}'
        for campo, (vecchio_valore, nuovo_valore) in sovrascritti.items())
    nota_lettura += ('<p class="nota avviso" style="margin-top:8px">Quello che hai '
                     f'scritto non coincideva con i campi, e ho seguito il testo — '
                     f'{elenco}. Se preferisci i campi, torna indietro e svuota il '
                     'riquadro di testo.</p>')

st.markdown(
    f'<div class="eroe"><div class="occhiello">La tua richiesta</div>'
    f'{citazione}'
    f'<div class="pillole" style="margin-top:14px">{pillole_profilo}</div>'
    f'{nota_lettura}</div>', unsafe_allow_html=True)

# --- calcolo ---
p = punteggio(catalogo, profilo)
stima = StimaAdattiva(profilo.capacita_stimata)
for indice in st.session_state.scartati:
    stima.aggiorna(float(catalogo["volume_settimanale"].iloc[indice]), accettato=False)

# La penalita' sul carico entra solo se gli scarti hanno mosso la stima: altrimenti
# il riordino farebbe credere a un apprendimento che non c'e'.
ha_imparato = abs(stima.capacita - profilo.capacita_stimata) > 0.05
if ha_imparato:
    # stessa curva e stesso peso dell'esperimento, presi da src/adaptive.py
    correzione = voto_sostenibilita(catalogo["volume_settimanale"].values
                                    / max(stima.capacita, 1e-6))
    p = p * ((1 - PESO_CORREZIONE) + PESO_CORREZIONE * correzione)

# ordina() come nella valutazione, non l'ordine interno di pandas.
ordinati = (p.iloc[ordina(p.values)]
            .drop(index=list(st.session_state.scartati), errors="ignore").dropna())
if ordinati.empty:
    st.warning("Con questa attrezzatura non trovo programmi adatti. "
               "Prova a indicarne una diversa.")
    st.stop()

# --- quanti ne ha trovati ---
# "adatti" vuol dire nessuna componente sotto la soglia. Il conto viene da
# rilevanza() stessa, cosi' segue la stessa regola delle schede. Gli scartati non
# contano.
ancora_in_lista = ~catalogo.index.isin(st.session_state.scartati)
fattibili = p.notna() & ancora_in_lista
gravemente_lontani = rilevanza(catalogo, profilo, dettaglio=True)["componenti_sotto_soglia"]
# serve per dire "nessun compromesso": tutte le componenti a 1
componenti_rilevanza = componenti(catalogo, profilo)
e_adatto = fattibili & (gravemente_lontani == 0)
adatti = int(e_adatto.sum())
compatibili = int(fattibili.sum())

# --- cosa e' cambiato dopo lo scarto ---
# Senza questa fascia l'adattamento non si vede: sembrerebbe un filtro qualunque.
effetto = st.session_state.effetto
nuovi_in_classifica = set()
if effetto:
    prima = effetto["capacita_prima"]
    dopo = stima.capacita
    if dopo < prima - 0.05:
        classifica_ora = [i for i, _ in ordinati.head(3).items()]
        nuovi_in_classifica = {i for i in classifica_ora if i not in effetto["top_prima"]}
        quanti_nuovi = len(nuovi_in_classifica)
        if quanti_nuovi == 0:
            coda = "I primi tre restano gli stessi: erano gia' adatti al tuo carico."
        elif quanti_nuovi == 1:
            coda = "Un programma nuovo e' entrato fra i primi tre."
        else:
            coda = f"{quanti_nuovi} programmi nuovi sono entrati fra i primi tre."
        larghezza = max(6.0, 100.0 * dopo / max(prima, 1e-6))
        st.markdown(
            f'<div class="fascia-adattiva">'
            f'<div class="capito">Ho capito.</div>'
            f'<div class="dettaglio">&ldquo;{escape(titolo_da_mostrare(effetto["titolo"]))}&rdquo; era piu\' carico '
            f'di quanto stimassi tu potessi reggere, quindi ho abbassato la stima '
            f'e ho rifatto la classifica. {coda}</div>'
            f'<div class="numeri">'
            f'<span class="prima">{prima:.0f}</span>'
            f'<span class="freccia">&rarr;</span>'
            f'<span class="dopo">{dopo:.0f}</span>'
            f'<span class="unita">serie a settimana che me la sento di proporti</span>'
            f'</div>'
            f'<div class="barra-stima"><i style="width:{larghezza:.0f}%"></i>'
            f'<u style="left:calc(100% - 2px)"></u></div>'
            f'</div>', unsafe_allow_html=True)
    else:
        # la stima non si e' mossa: non si promette il contrario
        st.markdown(
            f'<p class="nota tolto">Ho tolto &ldquo;{escape(titolo_da_mostrare(effetto["titolo"]))}&rdquo; dalla lista. '
            f'Era gia\' leggero rispetto a quello che stimavo tu potessi reggere, '
            f'quindi non ho cambiato la stima.</p>', unsafe_allow_html=True)

    angolo, spazio = st.columns([1, 5])
    with angolo:
        if st.button("Annulla", use_container_width=True):
            if st.session_state.scartati:
                st.session_state.scartati.pop()
            st.session_state.effetto = None
            st.rerun()

st.markdown('<div class="titolo-sezione">I programmi piu\' adatti a te</div>',
            unsafe_allow_html=True)
totale_programmi = f"{len(catalogo):,}".replace(",", ".")
compatibili_it = f"{compatibili:,}".replace(",", ".")
st.markdown(
    f'<p class="nota">Ho confrontato <b>{totale_programmi}</b> programmi: '
    f'<b>{compatibili_it}</b> si possono fare con la tua attrezzatura e '
    f'<b>{adatti}</b> {"rispetta" if adatti == 1 else "rispettano"} anche obiettivo, '
    f'livello, tempo e giorni'
    + ((' (senza quello che hai scartato)' if len(st.session_state.scartati) == 1
        else f' (senza i {len(st.session_state.scartati)} che hai scartato)')
       if st.session_state.scartati else '') +
    '. Qui sotto i migliori.</p>', unsafe_allow_html=True)
# Le prime tre sono la classifica valutata. "Mostrane altri 3" aggiunge solo
# programmi adatti, in ordine di punteggio, finche' ce ne sono: fermarsi prima
# nasconderebbe programmi adatti.
prime = list(ordinati.head(3).items())
altri_adatti = [(i, v) for i, v in ordinati.iloc[3:].items() if e_adatto[i]]
disponibili = len(prime) + len(altri_adatti)
da_mostrare = min(st.session_state.quanti, disponibili)
proposte = (prime + altri_adatti)[:da_mostrare]

for inizio_riga in range(0, len(proposte), 3):
    blocco = proposte[inizio_riga:inizio_riga + 3]
    colonne = st.columns(3, gap="medium")

    for scostamento, ((indice, _), colonna) in enumerate(zip(blocco, colonne)):
        posizione = inizio_riga + scostamento
        riga = catalogo.loc[indice]
        livello_riga = (int(riga["livello_min"]) if pd.notna(riga["livello_min"])
                        else profilo.livello_num)
        # tutte le schede hanno la stessa struttura: le righe vuote restano invisibili
        frase_enfasi = punto_enfasi(riga, profilo)
        forza = ([frase_enfasi] if frase_enfasi else []) + punti_di_forza(riga)
        forza = forza[:2]
        tutti_i_punti = punti_di_attenzione(riga, profilo, capacita=stima.capacita)
        attenzione = tutti_i_punti[:2]
        if not attenzione:
            # "nessun compromesso" solo se ogni componente e' piena
            piene = bool((componenti_rilevanza.loc[indice] >= 1 - 1e-9).all())
            attenzione = ["nessun compromesso rispetto a quello che hai chiesto" if piene
                          else "si discosta solo di poco da quello che hai chiesto"]

        # segno le schede nuove dopo lo scarto, se no non si capisce cosa e' cambiato
        if indice in nuovi_in_classifica:
            classe_ordine, testo_ordine = "ordine nuovo", "NUOVO DOPO LA TUA SCELTA"
        else:
            classe_ordine = "ordine" if posizione == 0 else "ordine neutro"
            # dopo le prime tre il numero della scheda non e' la posizione: si dice
            # quella vera
            testo_ordine = (etichetta_ordine(posizione) if posizione < 3 else
                            f"{ordinati.index.get_loc(indice) + 1}° IN CLASSIFICA")

        with colonna:
            classi = "scheda affiancata" + (" prima" if posizione == 0 else "")
            blocchi = [
                f'<div class="{classi}" style="animation-delay:{(posizione % 3) * 0.07:.2f}s">',
                f'<div class="{classe_ordine}">{testo_ordine}</div>',
                f'<div class="titolo-scheda">{escape(titolo_da_mostrare(riga["title"]))}</div>',
                f'<div class="sottotitolo">Per {DESCRIZIONE_LIVELLO[livello_riga]}</div>',
                '<div class="pillole">',
                pillola(f'{riga["settimane"]:.0f}', "sett."),
                pillola(f'{riga["giorni_settimana"]:.0f}', "a sett."),
                pillola(f'{riga["time_per_workout"]:.0f}\'', "a seduta"),
                '</div>',
            ]
            blocchi.append('<div class="etichetta-gruppo">Perche\' te lo consiglio</div>')
            blocchi += [punto(f, "si") for f in forza]
            blocchi += ['<div class="punto vuoto"><span class="segno si">v</span>'
                        '<span>riga segnaposto</span></div>'] * (2 - len(forza))
            blocchi.append('<div class="etichetta-gruppo">Da sapere'
                           + (f' (i 2 piu\' importanti di {len(tutti_i_punti)})'
                              if len(tutti_i_punti) > 2 else '') + '</div>')
            blocchi += [punto(a, "attenzione") for a in attenzione]
            blocchi += ['<div class="punto vuoto"><span class="segno attenzione">!</span>'
                        '<span>riga segnaposto</span></div>'] * (2 - len(attenzione))
            blocchi.append('</div>')
            st.markdown("".join(blocchi), unsafe_allow_html=True)

            if st.button("Vedi la scheda", key=f"apri_{indice}", type="primary",
                         use_container_width=True):
                st.session_state.aperto = riga["title"]
                st.rerun()

            # L'etichetta dice cio' che il sistema sa usare: il rifiuto abbassa la
            # stima del carico. Con "non fa per me" chi scarta per gusto insegnerebbe la
            # cosa sbagliata.
            if st.button("Troppo impegnativo", key=f"scarta_{indice}",
                         use_container_width=True):
                st.session_state.effetto = {
                    "capacita_prima": float(stima.capacita),
                    "top_prima": [i for i, _ in proposte[:3]],
                    "titolo": riga["title"],
                }
                st.session_state.scartati.append(indice)
                st.rerun()

# --- vederne altri ---
if disponibili > da_mostrare:
    restanti = disponibili - da_mostrare
    st.write("")
    spazio_sinistra, centro, spazio_destra = st.columns([1, 1.4, 1])
    with centro:
        if st.button(f"Mostrane altri 3  (adatti ancora da vedere: {restanti})",
                     use_container_width=True):
            st.session_state.quanti = da_mostrare + 3
            st.rerun()

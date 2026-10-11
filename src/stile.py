"""Foglio di stile della demo per l'utente finale.

Tutti i colori passano da queste costanti, cosi' cambiarne uno lo cambia ovunque.
Il tema di Streamlit (.streamlit/config.toml) li ripete: controlla_tema() verifica
all'avvio che coincidano.
"""
import re
from html import escape
from pathlib import Path

ACCENTO = "#D4FF57"
ACCENTO_CHIARO = "#E2FF7A"  # al passaggio del mouse
ACCENTO_FINE = "#B9F53C"  # fine della sfumatura dei pulsanti
ACCENTO_SCURO = "#0A0D12"  # testo sopra l'accento
BIANCO = "#FFFFFF"
FONDO = "#080A0F"
SUPERFICIE = "#11161F"  # usato solo dal tema
BORDO = "rgba(255,255,255,0.08)"  # = _rgba(BIANCO, 0.08)
BORDO_CAMPO = "rgba(255,255,255,0.40)"  # = _rgba(BIANCO, 0.40), abbastanza visibile da capire dove scrivere
TESTO = "#F2F5F9"
TESTO_SECONDARIO = "#C9D4E0"
TESTO_TENUE = "#A3AFBE"  # piu' scuro non basta per il contrasto sotto il bagliore
VERDE = "#5BE49B"
AMBRA = "#FFC65C"


def _rgba(colore: str, alfa: float) -> str:
    """Restituisce una costante di colore con trasparenza, per esempio _rgba(ACCENTO, 0.2)."""
    c = colore.lstrip("#")
    return f"rgba({int(c[0:2], 16)},{int(c[2:4], 16)},{int(c[4:6], 16)},{alfa})"


def controlla_tema(percorso=Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"):
    """Controlla che il tema di Streamlit usi i colori di questo file."""
    if not percorso.exists():
        return
    testo = percorso.read_text()
    attesi = {"primaryColor": ACCENTO, "backgroundColor": FONDO,
              "secondaryBackgroundColor": SUPERFICIE, "textColor": TESTO}
    for chiave, valore in attesi.items():
        trovato = re.search(rf'^{chiave}\s*=\s*"([^"]+)"', testo, re.M)
        # un errore e non un assert: con python -O gli assert spariscono
        if not (trovato and trovato.group(1).upper() == valore.upper()):
            raise RuntimeError(f"{percorso.name}: {chiave} non coincide con "
                               f"src/stile.py ({valore})")


controlla_tema()

CSS = f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

  .stApp {{
    background:
      radial-gradient(900px 500px at 12% -8%, {_rgba(ACCENTO, 0.08)}, transparent 60%),
      radial-gradient(800px 500px at 92% 4%, {_rgba(VERDE, 0.08)}, transparent 55%),
      {FONDO};
    font-family: 'Inter', system-ui, sans-serif;
  }}
  #MainMenu, footer, header {{ visibility: hidden; }}
  .block-container {{
    padding-top: 1.6rem; padding-bottom: 4rem;
    max-width: 100% !important; width: 100% !important;
    padding-left: clamp(1rem, 4vw, 4rem); padding-right: clamp(1rem, 4vw, 4rem);
  }}
  /* su schermi molto larghi il contenuto non si sgrana: si allarga, non si stira */
  @media (min-width: 1700px) {{
    .block-container {{ padding-left: 8vw; padding-right: 8vw; }}
  }}

  h1, h2, h3, h4, .titolo-scheda, .eroe h1 {{
    font-family: 'Space Grotesk', system-ui, sans-serif; letter-spacing: -0.02em;
  }}

  @keyframes entrata {{
    from {{ opacity: 0; transform: translateY(14px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
  }}
  @media (prefers-reduced-motion: reduce) {{
    * {{ animation: none !important; transition: none !important; }}
  }}

  /* ---------------------------------------------------------- intestazione */
  .eroe {{
    position: relative; overflow: hidden;
    background: linear-gradient(145deg, {_rgba(BIANCO, 0.055)}, {_rgba(BIANCO, 0.015)});
    border: 1px solid {BORDO}; border-radius: 26px;
    padding: clamp(30px, 3vw, 52px) clamp(28px, 3vw, 54px); margin-bottom: 26px;
    backdrop-filter: blur(18px);
    animation: entrata 0.5s ease both;
  }}
  .eroe::after {{
    content: ""; position: absolute; top: -60%; right: -10%;
    width: 320px; height: 320px; border-radius: 50%;
    background: radial-gradient(circle, {_rgba(ACCENTO, 0.08)}, transparent 65%);
    pointer-events: none;
  }}
  .occhiello {{
    display: inline-flex; align-items: center; gap: 7px;
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.14em;
    text-transform: uppercase; color: {ACCENTO};
    border: 1px solid {_rgba(ACCENTO, 0.28)}; border-radius: 999px;
    padding: 5px 12px; margin-bottom: 18px;
  }}
  .eroe h1 {{
    font-size: clamp(2.4rem, 3.4vw, 3.6rem); font-weight: 700; line-height: 1.06;
    margin: 0 0 14px 0; color: {TESTO};
  }}
  .eroe h1 em {{
    font-style: normal;
    background: linear-gradient(100deg, {ACCENTO}, {VERDE});
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  }}
  .eroe p {{ color: {TESTO_TENUE}; font-size: clamp(1rem, 1.15vw, 1.25rem);
             margin: 0; max-width: 72ch; line-height: 1.55; }}

  /* ------------------------------------------------------ schede programmi */
  .scheda {{
    position: relative;
    background: linear-gradient(160deg, {_rgba(BIANCO, 0.045)}, {_rgba(BIANCO, 0.012)});
    border: 1px solid {BORDO}; border-radius: 22px;
    padding: 24px 26px 20px 26px; margin-bottom: 12px;
    backdrop-filter: blur(14px);
    transition: transform .22s ease, border-color .22s ease, box-shadow .22s ease;
    animation: entrata 0.5s ease both;
  }}
  div[data-testid="stColumn"] {{ display: flex; flex-direction: column; }}
  div[data-testid="stColumn"] > div {{ height: 100%; display: flex; flex-direction: column; }}
  div[data-testid="stColumn"] div[data-testid="stVerticalBlock"] {{ height: 100%; }}
  /* stessa altezza per tutte: cosi' anche i pulsanti sotto restano allineati */
  div[data-testid="stHorizontalBlock"] {{ align-items: stretch; }}
  .scheda.affiancata {{
    display: flex; flex-direction: column; flex: 1 1 auto;
    height: 100%; min-height: 470px; margin-bottom: 10px;
  }}
  .scheda.affiancata .titolo-scheda {{
    min-height: 3.4rem; display: flex; align-items: flex-start;
  }}
  .scheda.affiancata .sottotitolo {{ min-height: 1.4rem; }}
  .scheda.affiancata .pillole {{ margin-top: auto; }}
  /* le pillole su una riga sola dentro le schede affiancate */
  .scheda.affiancata .pillola {{ flex: 1 1 0; min-width: 0; padding: 10px 8px; text-align: center; }}
  .scheda.affiancata .pillola b {{ font-size: 1.05rem; }}
  .scheda.affiancata .pillola span {{ font-size: 0.58rem; letter-spacing: 0.04em; }}
  .scheda:hover {{
    transform: translateY(-2px);
    border-color: {_rgba(ACCENTO, 0.35)};
    box-shadow: 0 18px 44px -24px {_rgba(ACCENTO, 0.45)};
  }}
  /* il programma piu' adatto: stessa identica scheda, solo il bordo cambia colore */
  .scheda.prima {{ border-color: {ACCENTO}; }}
  .scheda.prima:hover {{ border-color: {ACCENTO}; }}
  .ordine {{
    display: inline-block; font-size: 0.66rem; font-weight: 700;
    letter-spacing: 0.13em; text-transform: uppercase;
    padding: 5px 11px; border-radius: 999px; margin-bottom: 14px;
    background: {_rgba(ACCENTO, 0.12)}; color: {ACCENTO};
    border: 1px solid {_rgba(ACCENTO, 0.25)};
  }}
  .ordine.neutro {{
    background: {_rgba(BIANCO, 0.05)}; color: {TESTO_TENUE};
    border-color: {BORDO};
  }}
  .titolo-scheda {{
    font-size: 1.38rem; font-weight: 600; color: {TESTO};
    margin: 0 0 6px 0; line-height: 1.25;
  }}
  .sottotitolo {{ color: {TESTO_TENUE}; font-size: 0.94rem; margin-bottom: 18px; }}

  /* --------------------------------------------------------------- pillole */
  .pillole {{ display: flex; gap: 9px; flex-wrap: wrap; margin-bottom: 18px; }}
  .pillola {{
    flex: 1 1 88px; background: {_rgba(BIANCO, 0.03)};
    border: 1px solid {BORDO}; border-radius: 14px; padding: 11px 13px;
    transition: border-color .2s ease;
  }}
  .pillola:hover {{ border-color: {_rgba(ACCENTO, 0.3)}; }}
  .pillola b {{
    display: block; color: {TESTO}; font-size: 1.22rem; font-weight: 600;
    font-family: 'Space Grotesk', sans-serif; font-variant-numeric: tabular-nums;
    line-height: 1.1;
  }}
  .pillola span {{
    color: {TESTO_TENUE}; font-size: 0.66rem; text-transform: uppercase;
    letter-spacing: 0.06em; font-weight: 500;
    display: block; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }}

  /* ---------------------------------------------------------- punti chiave */
  .etichetta-gruppo {{
    color: {TESTO_TENUE}; font-size: 0.68rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.12em; margin: 16px 0 9px 0;
  }}
  .punto {{
    display: flex; gap: 11px; align-items: flex-start; margin-bottom: 8px;
    color: {TESTO_SECONDARIO}; font-size: 0.93rem; line-height: 1.5;
  }}
  /* riga segnaposto: occupa lo spazio ma non si vede, cosi' tutte le schede
     hanno lo stesso numero di righe e quindi la stessa altezza */
  .punto.vuoto {{ visibility: hidden; }}
  /* ogni riga occupa almeno due righe di testo, cosi' le schede restano allineate;
     ma una frase piu' lunga NON viene troncata. Prima un line-clamp a due righe
     tagliava fino al 58% delle frasi "Da sapere" su uno schermo da 1024 px: le
     frasi oneste c'erano, e poi sparivano. Meglio una scheda un po' piu' alta. */
  .scheda.affiancata .punto {{ min-height: 2.9rem; }}
  .segno {{
    flex: 0 0 auto; width: 19px; height: 19px; border-radius: 6px;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: 800; margin-top: 2px;
  }}
  .segno.si {{ background: {_rgba(VERDE, 0.14)}; color: {VERDE};
               box-shadow: inset 0 0 0 1px {_rgba(VERDE, 0.25)}; }}
  .segno.attenzione {{ background: {_rgba(AMBRA, 0.14)}; color: {AMBRA};
                       box-shadow: inset 0 0 0 1px {_rgba(AMBRA, 0.25)}; }}

  /* ------------------------------------------------- giorni di allenamento */
  .giorno {{
    background: linear-gradient(160deg, {_rgba(BIANCO, 0.04)}, {_rgba(BIANCO, 0.01)});
    border: 1px solid {BORDO}; border-radius: 20px;
    padding: 20px 22px; margin-bottom: 12px; backdrop-filter: blur(14px);
    animation: entrata 0.45s ease both;
  }}
  .intestazione-giorno {{
    display: flex; align-items: baseline; justify-content: space-between;
    margin-bottom: 14px; padding-bottom: 12px; border-bottom: 1px solid {BORDO};
  }}
  .intestazione-giorno h4 {{ margin: 0; color: {TESTO}; font-size: 1.05rem; font-weight: 600; }}
  .intestazione-giorno .conteggio {{ color: {TESTO_TENUE}; font-size: 0.78rem; }}
  .esercizio {{
    display: flex; justify-content: space-between; align-items: center; gap: 14px;
    padding: 10px 10px; margin: 0 -10px; border-radius: 10px; transition: background .15s ease;
  }}
  .esercizio:hover {{ background: {_rgba(BIANCO, 0.035)}; }}
  .esercizio .indice {{
    flex: 0 0 auto; width: 22px; color: {TESTO_TENUE}; font-size: 0.78rem;
    font-variant-numeric: tabular-nums;
  }}
  .esercizio .nome {{ flex: 1; color: {TESTO}; font-size: 0.92rem; }}
  .esercizio .serie {{
    color: {ACCENTO}; font-weight: 600; font-size: 0.92rem; white-space: nowrap;
    font-family: 'Space Grotesk', sans-serif; font-variant-numeric: tabular-nums;
  }}
  .esercizio .rpe {{
    color: {TESTO_TENUE}; font-weight: 500; font-size: 0.78rem; margin-left: 6px;
  }}

  /* -------------------------------------------------------------- pulsanti */
  .stButton > button {{
    border-radius: 13px; font-weight: 600; font-family: 'Inter', sans-serif;
    border: 1px solid {BORDO}; background: {_rgba(BIANCO, 0.04)};
    color: {TESTO}; padding: 0.6rem 1.1rem; transition: all .18s ease;
  }}
  .stButton > button p {{ font-weight: 600; margin: 0; }}
  .stButton > button:hover {{
    border-color: {_rgba(ACCENTO, 0.5)}; background: {_rgba(ACCENTO, 0.07)};
    color: {ACCENTO}; transform: translateY(-1px);
  }}
  .stButton > button:hover p {{ color: {ACCENTO}; }}

  /* pulsante d'accento: testo sempre scuro, in ogni stato e in ogni versione */
  button[kind*="primary" i],
  button[data-testid*="primary" i],
  div[data-testid="stForm"] button[kind*="Form" i] {{
    background: linear-gradient(100deg, {ACCENTO}, {ACCENTO_FINE}) !important;
    border: none !important; box-shadow: 0 10px 30px -14px {_rgba(ACCENTO, 0.75)};
  }}
  button[kind*="primary" i], button[kind*="primary" i] *,
  button[data-testid*="primary" i], button[data-testid*="primary" i] *,
  div[data-testid="stForm"] button[kind*="Form" i],
  div[data-testid="stForm"] button[kind*="Form" i] * {{
    color: {ACCENTO_SCURO} !important; fill: {ACCENTO_SCURO} !important;
    -webkit-text-fill-color: {ACCENTO_SCURO} !important;
  }}
  button[kind*="primary" i]:hover,
  button[data-testid*="primary" i]:hover,
  div[data-testid="stForm"] button[kind*="Form" i]:hover {{
    background: linear-gradient(100deg, {ACCENTO_CHIARO}, {ACCENTO}) !important;
    transform: translateY(-1px);
  }}
  .stButton > button:focus:not(:active) {{ border-color: {ACCENTO}; }}

  /* ----------------------------------------------------------- modulo e campi */
  div[data-testid="stForm"] {{
    background: linear-gradient(160deg, {_rgba(BIANCO, 0.045)}, {_rgba(BIANCO, 0.012)});
    border: 1px solid {BORDO}; border-radius: 26px;
    padding: clamp(24px, 2.6vw, 44px);
    backdrop-filter: blur(16px); animation: entrata 0.55s ease both;
  }}
  /* campi del modulo piu' generosi */
  div[data-testid="stForm"] .stTextArea textarea {{ font-size: 1.02rem; line-height: 1.5; }}
  div[data-testid="stForm"] label {{ font-size: 0.92rem !important; white-space: normal; }}
  div[data-testid="stForm"] div[data-baseweb="select"] > div {{
    min-height: 46px; font-size: 0.98rem;
  }}
  div[data-testid="stForm"] button {{ font-size: 1.02rem; padding: 0.8rem 1.2rem; }}
  /* i menu a tendina in Streamlit 1.64 sono ComboBox di React Aria: il selettore
     baseweb di prima non prendeva niente, e il loro bordo era invisibile (1,00:1).
     Selettore letto dal DOM nel browser. */
  .stTextArea textarea, div[data-baseweb="select"] > div,
  [data-testid="stSelectbox"] div[role="group"] {{
    background: {_rgba(BIANCO, 0.03)} !important; border-radius: 13px !important;
    border: 1px solid {BORDO_CAMPO} !important; color: {TESTO} !important;
  }}
  .stTextArea textarea:focus {{ border-color: {_rgba(ACCENTO, 0.45)} !important; }}
  /* Pillole delle zone non scelte: con il bordo di Streamlit non si vedeva dove finiva
     ognuna. Al passaggio del mouse Streamlit schiarisce lo sfondo, il bordo resta.
     (selettore letto dal DOM nel browser) */
  [data-testid="stButtonGroup"] button[data-variant="pills"][aria-pressed="false"] {{
    border-color: {BORDO_CAMPO} !important;
  }}

  .eroe .citazione, .citazione {{   /* .eroe p vinceva: era del colore tenue */
    color: {TESTO}; font-size: 1.12rem; line-height: 1.5; font-style: italic;
    margin: 0; padding-left: 14px; border-left: 2px solid {ACCENTO}; max-width: 70ch;
  }}

  .titolo-sezione {{
    font-family: 'Space Grotesk', sans-serif; font-size: 1.35rem; font-weight: 600;
    color: {TESTO}; margin: 6px 0 4px 0;
  }}
  /* Avvisi. Prima uno sfondo neutro era forzato su TUTTI, anche sugli errori: la
     domanda sull'attrezzatura, l'unica interazione di sicurezza, perdeva il colore.
     Ora solo quelli informativi sono neutri; errori e avvertimenti (role="alert",
     dal sorgente di Streamlit 1.64) hanno il bordo ambra. */
  .stAlert {{ border-radius: 14px; }}
  .stAlert:has([data-testid="stAlertContentInfo"]) {{
    border: 1px solid {BORDO}; background: {_rgba(BIANCO, 0.04)} !important;
  }}
  [data-testid="stAlertContainer"][role="alert"] {{
    border: 1px solid {AMBRA}; border-radius: 14px;
  }}
  /* l'avviso "il testo non coincideva con i campi": classe usata, prima mai definita */
  .nota.avviso {{
    color: {TESTO}; padding-left: 12px; border-left: 2px solid {AMBRA};
  }}
  /* ------------------------------------------------- fascia dell'adattamento
     Compare quando uno scarto ha davvero spostato la stima della capacita'.
     Deve rendere visibile un cambiamento che altrimenti l'utente non vedrebbe:
     il numero che scende, la barra che si accorcia, le schede nuove marcate. */
  .fascia-adattiva {{
    background: linear-gradient(120deg, {_rgba(ACCENTO, 0.10)}, {_rgba(BIANCO, 0.015)});
    border: 1px solid {_rgba(ACCENTO, 0.28)}; border-radius: 20px;
    padding: 20px 24px; margin: 4px 0 20px 0; backdrop-filter: blur(14px);
    animation: entrata 0.45s ease both;
  }}
  .fascia-adattiva .capito {{
    font-family: 'Space Grotesk', sans-serif; font-size: 1.12rem;
    font-weight: 600; color: {TESTO}; margin-bottom: 6px;
  }}
  .fascia-adattiva .dettaglio {{ color: {TESTO_TENUE}; font-size: 0.95rem; line-height: 1.55; }}
  .fascia-adattiva .numeri {{
    display: flex; align-items: baseline; gap: 10px; margin: 14px 0 6px 0;
    flex-wrap: wrap;
  }}
  .fascia-adattiva .prima {{
    color: {TESTO_TENUE}; font-size: 1.15rem; text-decoration: line-through;
    text-decoration-thickness: 1px;
  }}
  .fascia-adattiva .freccia {{ color: {TESTO_TENUE}; font-size: 1.05rem; }}
  .fascia-adattiva .dopo {{
    color: {ACCENTO}; font-size: 1.8rem; font-weight: 700;
    font-family: 'Space Grotesk', sans-serif; line-height: 1;
  }}
  .fascia-adattiva .unita {{ color: {TESTO_TENUE}; font-size: 0.9rem; }}
  /* la barra: la parte piena e' la stima nuova, il tratteggio quella vecchia */
  .barra-stima {{
    position: relative; height: 8px; border-radius: 999px;
    background: {_rgba(BIANCO, 0.06)}; border: 1px solid {BORDO};
    max-width: 460px; margin-top: 4px; overflow: hidden;
  }}
  .barra-stima i {{
    position: absolute; left: 0; top: 0; bottom: 0; display: block;
    background: linear-gradient(100deg, {ACCENTO}, {VERDE});
    border-radius: 999px; transition: width .5s ease;
  }}
  .barra-stima u {{
    position: absolute; top: -3px; bottom: -3px; width: 2px;
    background: {_rgba(BIANCO, 0.45)};
  }}
  /* etichetta sulle schede entrate in classifica dopo lo scarto */
  .ordine.nuovo {{
    background: {_rgba(ACCENTO, 0.16)}; color: {ACCENTO};
    border-color: {_rgba(ACCENTO, 0.4)};
  }}
  .nota.tolto {{
    color: {TESTO_TENUE}; font-size: 0.92rem; margin: 0 0 16px 0;
    padding-left: 12px; border-left: 2px solid {BORDO};
  }}
</style>
"""


# Le funzioni fanno l'escape di cio' che ricevono: i nomi degli esercizi vengono dal
# dataset, e la sicurezza non deve dipendere dai dati.
def pillola(valore, etichetta) -> str:
    return (f'<div class="pillola"><b>{escape(str(valore), quote=False)}</b>'
            f'<span>{escape(str(etichetta), quote=False)}</span></div>')


def punto(testo, tipo="si") -> str:
    simbolo = "✓" if tipo == "si" else "!"
    return (f'<div class="punto"><span class="segno {tipo}">{simbolo}</span>'
            f'<span>{escape(str(testo), quote=False)}</span></div>')


def riga_esercizio(indice, nome, serie) -> str:
    serie = str(serie)
    if "RPE" in serie:
        carico, rpe = serie.split("RPE", 1)
        serie_html = (f'{escape(carico.strip(), quote=False)}'
                      f'<span class="rpe">RPE {escape(rpe.strip(), quote=False)}</span>')
    else:
        serie_html = escape(serie, quote=False)
    return (f'<div class="esercizio"><span class="indice">{escape(str(indice), quote=False)}</span>'
            f'<span class="nome">{escape(str(nome), quote=False)}</span>'
            f'<span class="serie">{serie_html}</span></div>')

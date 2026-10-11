"""Stile comune delle figure del progetto.

Virgola decimale, testo in italiano, nomi leggibili e una sola funzione per salvare.
Dove una categoria conta, oltre al colore c'e' un tratteggio o una forma diversa, cosi'
la figura si legge anche in bianco e nero.
"""
import ast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from src.data_loading import OUT_FIG

DPI = 200

# Blu per cio' che aiuta, rosso per cio' che danneggia, verde per cio' che serve,
# grigio per cio' che non si distingue da zero.
BLU = "#2a78d6"
BLU_CHIARO = "#86b6ef"
ROSSO = "#d64541"
VERDE = "#2e8b57"
GRIGIO = "#a3a3a3"
ARANCIO = "#e08a1e"
INCHIOSTRO = "#0b0b0b"
INCHIOSTRO_2 = "#52514e"
GRIGLIA = "#dedcd6"

NOMI_INDICATORI = {
    "q_volume": "volume",
    "q_frequenza": "frequenza",
    "q_progressione": "progressione",
    "q_equilibrio": "equilibrio muscolare",
    "q_varieta": "varietà",
    "q_coerenza": "coerenza con l'obiettivo",
    "qualita": "qualità complessiva",
    # l'ablation usa i nomi senza prefisso
    "volume": "volume", "frequenza": "frequenza", "progressione": "progressione",
    "equilibrio": "equilibrio muscolare", "varieta": "varietà",
    "coerenza": "coerenza con l'obiettivo",
}

NOMI_SISTEMI = {
    "casuale": "casuale",
    "solo metadati": "solo metadati",
    "solo testo (TF-IDF)": "solo testo (TF-IDF)",
    "solo testo (semantico LSA)": "solo testo (LSA)",
    "solo testo (embeddings)": "solo testo (embeddings)",
    "rilevanza graduale": "rilevanza graduale",
    "completo (rilevanza + qualita)": "completo (rilevanza + qualità)",
}

NOMI_SCENARI = {
    "riferimento": "riferimento",
    "capacita' -30%": "capacità −30%",
    "capacita' +30%": "capacità +30%",
    "rumore alto": "rumore alto",
    "orari rigidi": "orari rigidi",
    "piu' utenti a casa": "più utenti a casa",
    "seme diverso": "seme diverso",
    "alta variabilita' personale": "alta variabilità personale",
}

NOMI_ATTREZZATURE = {
    "Dumbbell Only": "solo manubri",
    "At Home": "a casa",
    "Garage Gym": "garage attrezzato",
    "Full Gym": "palestra completa",
}


def applica_stile():
    """Imposta lo stile di matplotlib; va chiamata prima di creare le figure."""
    plt.rcParams.update({
        "font.size": 10,
        "text.color": INCHIOSTRO,
        "axes.labelcolor": INCHIOSTRO_2,
        "axes.titlecolor": INCHIOSTRO,
        "xtick.color": INCHIOSTRO_2,
        "ytick.color": INCHIOSTRO,
        "axes.edgecolor": GRIGLIA,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.formatter.use_locale": False,
        "hatch.linewidth": 0.8,
    })


def numero(x, cifre=3, segno=False, migliaia=False):
    """Scrive un numero nel formato italiano: 0,783 / +0,0115 / −0,058 / 2.699."""
    if migliaia:
        testo = f"{x:,.{cifre}f}".replace(",", "§").replace(".", ",").replace("§", ".")
    else:
        testo = f"{x:.{cifre}f}".replace(".", ",")
    if testo.lstrip("-").strip("0,.") == "":  # uno zero arrotondato non ha segno
        return testo.lstrip("-")
    if testo.startswith("-"):
        return "−" + testo[1:]
    return ("+" + testo) if segno else testo


def _formato_automatico(v, _pos=None):
    """Formatta una tacca senza cifre fisse: 0,2 / 0,75 / 1 / 1.000."""
    if abs(v) >= 1000:
        return numero(v, 0, migliaia=True)
    testo = f"{round(v, 6):g}"
    if "e" in testo:
        testo = f"{v:.6f}".rstrip("0").rstrip(".")
    testo = testo.replace(".", ",")
    return ("−" + testo[1:]) if testo.startswith("-") else testo


def virgola(ax, asse="both", cifre=None):
    """Mette la virgola decimale sulle tacche dell'asse (x, y o both)."""
    if cifre is None:
        formato = FuncFormatter(_formato_automatico)
    else:
        formato = FuncFormatter(lambda v, _pos: numero(v, cifre))
    if asse in ("x", "both"):
        ax.xaxis.set_major_formatter(formato)
    if asse in ("y", "both"):
        ax.yaxis.set_major_formatter(formato)


def pulisci(ax, griglia="x"):
    """Toglie la cornice superflua e mette una griglia tenue sotto i dati."""
    for lato in ("top", "right"):
        ax.spines[lato].set_visible(False)
    if griglia in ("x", "both"):
        ax.xaxis.grid(True, color=GRIGLIA, linewidth=0.8)
    if griglia in ("y", "both"):
        ax.yaxis.grid(True, color=GRIGLIA, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def intervallo(testo):
    """Legge un intervallo come '[0.780, 0.799]' e restituisce (inferiore, superiore)."""
    a, b = ast.literal_eval(str(testo).replace("+", ""))
    return float(a), float(b)


def errori(valori, intervalli):
    """Prepara le barre d'errore asimmetriche nel formato di matplotlib."""
    basso = [max(0.0, v - lo) for v, (lo, _) in zip(valori, intervalli)]
    alto = [max(0.0, hi - v) for v, (_, hi) in zip(valori, intervalli)]
    return [basso, alto]


def salva(fig, nome):
    """Salva la figura, con la stessa risoluzione per tutte."""
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG / f"{nome}.png", dpi=DPI, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  -> salvata outputs/figures/{nome}.png")

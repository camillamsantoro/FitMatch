"""12 - Sentence embeddings per la baseline testuale.

Usa lo stesso modello del notebook Kaggle (all-MiniLM-L6-v2). Serve torch, quindi
li calcolo una volta e li salvo. Si esegue nel .venv del progetto:
    python scripts/12_embeddings.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json
from src.data_loading import testo_programmi, impronta_testi
from src.simulator import genera_utenti
from src.recommenders import chiave_query

PROCESSED = ROOT / "data" / "processed"

# Fisso la revisione: col solo nome Hugging Face darebbe l'ultima versione, e un
# altro computer potrebbe avere embedding diversi senza accorgersene.
MODELLO = "sentence-transformers/all-MiniLM-L6-v2"
REVISIONE = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


def main():
    from sentence_transformers import SentenceTransformer  # import qui perche' e' pesante

    catalogo = pd.read_csv(PROCESSED / "catalogo_con_qualita.csv")
    # Se manca lo costruisce: 07 a sua volta ha bisogno di questi embedding.
    testi = testo_programmi(catalogo)

    modello = SentenceTransformer(MODELLO, revision=REVISIONE)
    print(f"Calcolo degli embedding per {len(testi)} programmi...")
    emb_programmi = modello.encode(testi, batch_size=64, show_progress_bar=True,
                                   normalize_embeddings=True)
    np.save(PROCESSED / "embeddings_programmi.npy", emb_programmi)

    utenti = genera_utenti(300)
    query = sorted({chiave_query(u.profilo_dichiarato()) for u in utenti})
    print(f"Calcolo degli embedding per {len(query)} profili distinti...")
    emb_query = modello.encode(query, batch_size=64, normalize_embeddings=True)
    np.save(PROCESSED / "embeddings_query.npy", emb_query)
    pd.Series(query, name="query").to_csv(PROCESSED / "embeddings_query_chiavi.csv", index=False)

    # Il manifesto dice da cosa sono nati i file, cosi' 07 lo controlla.
    import sentence_transformers, torch
    manifesto = {
        "modello": MODELLO, "revisione": REVISIONE,
        "sentence_transformers": sentence_transformers.__version__,
        "torch": torch.__version__,
        "programmi": len(testi), "impronta_testi": impronta_testi(testi),
        "query": len(query), "normalizzati": True,
    }
    (PROCESSED / "embeddings_manifesto.json").write_text(json.dumps(manifesto, indent=2))
    print("Fatto. Ora rieseguire: python scripts/07_valutazione.py")


if __name__ == "__main__":
    main()

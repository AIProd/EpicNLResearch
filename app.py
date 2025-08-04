import streamlit as st
import pandas as pd
import re
import spacy
from scispacy.linking import EntityLinker            # registers scispacy_linker
import pdfplumber                                     # (needed only if you later add PDF)

# ─────────── NLP loading ───────────
@st.cache_resource(show_spinner="Loading SciSpaCy model…")
def load_nlp():
    nlp = spacy.load("en_core_sci_sm")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer", first=True)
    if "scispacy_linker" not in nlp.pipe_names:
        nlp.add_pipe(
            "scispacy_linker",
            config={"resolve_abbreviations": True, "linker_name": "umls"}
        )
    return nlp

nlp = load_nlp()
linker = nlp.get_pipe("scispacy_linker")

# ─────────── UI ───────────
st.set_page_config("Pathology Search POC", layout="wide")
st.title("🔬 Pathology Report Search (CSV prototype)")

uploaded_csv = st.file_uploader("Upload pathology CSV (cols: patient_filename, text)", type="csv")

query = st.text_input("Enter keywords (comma-separated)", "bladder, recurrence")

if uploaded_csv and query.strip():
    df = pd.read_csv(uploaded_csv)
    keywords = [q.strip().lower() for q in query.split(",") if q.strip()]
    max_rows = st.slider("Rows to scan", 1, min(500, len(df)), 50)

    hits = []
    progress = st.progress(0.0, text="Processing…")
    for i, row in df.head(max_rows).iterrows():
        doc = nlp(row["text"])
        sent_hit = [s.text.strip() for s in doc.sents
                    if any(k in s.text.lower() for k in keywords)]
        if sent_hit:
            cuiset = set()
            for ent in doc.ents:
                for cui, _ in ent._.kb_ents:
                    cuiset.add(cui)
            hits.append({
                "patient_filename": row["patient_filename"],
                "matched_sentences": " | ".join(sent_hit),
                "CUIs": ", ".join(sorted(cuiset))
            })
        progress.progress((i + 1) / max_rows)
    progress.empty()

    if hits:
        st.success(f"Found {len(hits)} matching reports.")
        st.dataframe(pd.DataFrame(hits), use_container_width=True)
    else:
        st.warning("No matches found in the scanned rows.")

else:
    st.info("Upload a CSV and enter search keywords to begin.")

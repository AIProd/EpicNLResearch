import streamlit as st
import pandas as pd
import spacy
import pdfplumber
import tempfile
import re

# 📦 Load SciSpaCy + UMLS linker
@st.cache_resource(show_spinner="Loading SciSpaCy model...")
def load_nlp():
    nlp = spacy.load("en_core_sci_sm")
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer", first=True)
    if "entity_linker" not in nlp.pipe_names:
        nlp.add_pipe("entity_linker", config={
            "resolve_abbreviations": True,
            "name": "umls"
        })
    return nlp

nlp = load_nlp()
linker = nlp.get_pipe("entity_linker")

# 🔍 Keywords + regex
BLADDER_KWS = ["bladder", "urothelial", "urinary", "ureter", "transitional cell carcinoma"]
RECUR_KWS = ["recurrence", "recurrent", "metastasis", "residual", "local invasion", "tumor"]
REGEX_PATTERNS = [r"T[0-4]N[0-3]M[0-1]", r"pT[0-4]"]

# 📃 PDF to text
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        return "\n".join([p.extract_text() or "" for p in pdf.pages])

# 🔎 Extract hits
def sentence_hits(doc, keywords):
    return [s.text.strip() for s in doc.sents if any(k in s.text.lower() for k in keywords)]

def regex_hits(text, patterns):
    sentences = re.split(r"(?<=[.!?])\s+", text)
    compiled = [re.compile(p, re.I) for p in patterns]
    return [s for s in sentences if any(p.search(s) for p in compiled)]

def umls_table(doc):
    rows = []
    for ent in doc.ents:
        for cui, score in ent._.kb_ents:
            concept = linker.kb.cui_to_entity[cui]
            rows.append({
                "Text": ent.text,
                "CUI": cui,
                "Name": concept.preferred_name,
                "Type": ", ".join(concept.types),
                "Score": round(score, 3)
            })
    return pd.DataFrame(rows)

# 🖼️ Streamlit UI
st.set_page_config("Clinical NLP – Bladder Cancer", layout="wide")
st.title("🩺 Bladder Cancer NLP Report Scanner")
st.markdown("Upload a pathology **PDF** or **TCGA CSV** to scan for bladder cancer and recurrence mentions.")

upload_type = st.radio("Input source", ["PDF report", "TCGA CSV sample"])

if upload_type == "PDF report":
    pdf = st.file_uploader("📄 Upload PDF", type=["pdf"])
    if pdf:
        with st.spinner("Extracting text..."):
            text = extract_text_from_pdf(pdf)
        with st.spinner("Running NLP..."):
            doc = nlp(text)
        st.success("✅ Analysis complete.")

        st.subheader("🔎 Bladder Mentions")
        st.write(sentence_hits(doc, BLADDER_KWS) or "_None_")

        st.subheader("🔄 Recurrence Mentions")
        st.write(sentence_hits(doc, RECUR_KWS) or "_None_")

        st.subheader("📐 Regex Matches (e.g., TNM stage)")
        st.write(regex_hits(text, REGEX_PATTERNS) or "_None_")

        st.subheader("🧬 UMLS Concepts")
        df_umls = umls_table(doc)
        st.dataframe(df_umls if not df_umls.empty else pd.DataFrame([{"Info": "No concepts found"}]))

        with st.expander("📜 Full extracted text"):
            st.text(text)

elif upload_type == "TCGA CSV sample":
    csv_file = st.file_uploader("🧾 Upload `TCGA_Reports.csv`", type=["csv"])
    if csv_file:
        df = pd.read_csv(csv_file)
        st.write(df.head())

        n = st.slider("📄 How many reports to process?", min_value=1, max_value=20, value=5)
        results = []
        with st.spinner(f"Processing {n} reports..."):
            for _, row in df.head(n).iterrows():
                doc = nlp(row["text"])
                bladder = sentence_hits(doc, BLADDER_KWS)
                recur = sentence_hits(doc, RECUR_KWS)
                if bladder or recur:
                    results.append({
                        "patient_filename": row["patient_filename"],
                        "bladder_mentions": "; ".join(bladder),
                        "recurrence_mentions": "; ".join(recur)
                    })
        out_df = pd.DataFrame(results)
        st.subheader("🧾 TCGA Results")
        st.dataframe(out_df if not out_df.empty else pd.DataFrame([{"Info": "No matches found"}]))

import os
import streamlit as st
# This example uses the "openai" package's modern client. Adjust if needed.
try:
    from openai import OpenAI
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception:
    # Fallback to old openai package if available
    import openai as _openai
    openai_client = None
    _openai.api_key = os.getenv("OPENAI_API_KEY")

import numpy as np
import pdfplumber
import docx
import tempfile

def extract_text_from_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    return "\\n".join([p.text for p in doc.paragraphs])

def file_to_text(uploaded_file):
    suffix = uploaded_file.name.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tf:
        tf.write(uploaded_file.getbuffer())
        tmp_path = tf.name
    if suffix.endswith(".pdf"):
        return extract_text_from_pdf(tmp_path)
    elif suffix.endswith(".docx"):
        return extract_text_from_docx(tmp_path)
    else:
        uploaded_file.seek(0)
        return uploaded_file.getvalue().decode(errors="ignore")

def get_embedding(text):
    # Truncate text to reasonable length for embeddings API
    text = text[:32000]
    if openai_client:
        resp = openai_client.embeddings.create(model="text-embedding-3-small", input=text)
        return np.array(resp.data[0].embedding, dtype=np.float32)
    else:
        # old openai package
        resp = _openai.Embedding.create(model="text-embedding-3-small", input=text)
        return np.array(resp["data"][0]["embedding"], dtype=np.float32)

def cosine_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

st.set_page_config(page_title="Resume Screening Agent — Demo")
st.title("Resume Screening Agent — Demo")
st.markdown("Upload a Job Description and candidate resumes; the agent ranks resumes by relevance using OpenAI embeddings.")

jd_text = st.text_area("Paste Job Description here", height=250)
uploaded_files = st.file_uploader("Upload resumes (.pdf/.docx/.txt)", accept_multiple_files=True)

if st.button("Score Resumes"):
    if not jd_text or not uploaded_files:
        st.error("Provide JD and at least one resume.")
    else:
        with st.spinner("Processing..."):
            try:
                jd_emb = get_embedding(jd_text)
            except Exception as e:
                st.error(f"Embedding error: {e}")
                st.stop()
            rows = []
            for f in uploaded_files:
                try:
                    txt = file_to_text(f)
                    emb = get_embedding(txt)
                    score = cosine_sim(jd_emb, emb)
                except Exception as e:
                    txt = ""
                    score = 0.0
                # Ask LLM for a short reason (best-effort; may fail without key or quota)
                reason = "(No explanation generated)"
                try:
                    prompt = f"Job description:\\n{jd_text}\\n\\nResume text (truncated):\\n{txt[:2000]}\\n\\nIn 2 sentences, why is this resume a good or bad fit? Provide concise bullets."
                    if openai_client:
                        chat = openai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], max_tokens=140)
                        reason = chat.choices[0].message.content.strip()
                    else:
                        chat = _openai.ChatCompletion.create(model="gpt-4o-mini", messages=[{"role":"user","content":prompt}], max_tokens=140)
                        reason = chat.choices[0].message["content"].strip()
                except Exception:
                    pass
                rows.append({"filename": f.name, "score": score, "reason": reason})
            rows_sorted = sorted(rows, key=lambda r: r["score"], reverse=True)
            st.write("## Ranked candidates")
            for r in rows_sorted:
                st.markdown(f"**{r['filename']}** — score: {r['score']:.4f}")
                st.markdown(r["reason"])
            # CSV export
            import io, csv
            csv_io = io.StringIO()
            writer = csv.writer(csv_io)
            writer.writerow(["filename","score","reason"])
            for r in rows_sorted:
                writer.writerow([r["filename"], f"{r['score']:.6f}", r["reason"]])
            st.download_button("Download results CSV", data=csv_io.getvalue(), file_name="screening_results.csv", mime="text/csv")

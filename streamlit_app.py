from datetime import datetime
from pathlib import Path

import requests
import streamlit as st


st.set_page_config(page_title="Production RAG", page_icon="🔎", layout="wide")

st.markdown(
    """
<style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .small-muted {color: #9aa0a6; font-size: 0.9rem;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("🔎 Production RAG")
st.markdown("<div class='small-muted'>Upload, ingest, and ask questions with sources.</div>", unsafe_allow_html=True)

api_base = "http://127.0.0.1:8000"

# Layout: left for ingest, right for chat
left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.subheader("📥 Ingest")
    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
    if st.button("Upload + Ingest", key="btn_upload_ingest", use_container_width=True):
        if not uploaded_file:
            st.warning("Please upload a PDF first.")
        else:
            upload_dir = Path("uploaded_pdfs")
            upload_dir.mkdir(exist_ok=True)
            safe_name = uploaded_file.name.replace(" ", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_path = upload_dir / f"{timestamp}_{safe_name}"
            saved_path.write_bytes(uploaded_file.getbuffer())
            st.caption(f"Saved: {saved_path.name}")
            try:
                resp = requests.post(
                    f"{api_base}/ingest",
                    params={"pdf_path": str(saved_path.resolve())},
                    timeout=300,
                )
                if resp.ok:
                    st.success("Ingestion complete")
                    st.json(resp.json())
                else:
                    st.error(f"API error {resp.status_code}")
                    st.json(resp.json())
            except Exception as e:
                st.error(f"Request failed: {e}")

with right_col:
    st.subheader("💬 Chat")
    question = st.text_area(
        "Ask anything from indexed documents",
        value="Give me a short summary of the indexed document.",
        height=150,
    )
    if st.button("Ask RAG", use_container_width=True):
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                resp = requests.post(
                    f"{api_base}/ask",
                    params={"question": question},
                    timeout=300,
                )
                if resp.ok:
                    data = resp.json()
                    st.markdown("### Answer")
                    st.write(data.get("answer", ""))

                    st.markdown("### Sources")
                    sources = data.get("sources", [])
                    if not sources:
                        st.info("No sources returned.")
                    else:
                        st.json(sources)
                else:
                    st.error(f"API error {resp.status_code}")
                    st.json(resp.json())
            except Exception as e:
                st.error(f"Request failed: {e}")


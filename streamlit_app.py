import json
from pathlib import Path

import requests
import streamlit as st


st.set_page_config(page_title="RAG Assistant", page_icon=":books:", layout="wide")


def _api_url() -> str:
    return st.session_state.get("api_base_url", "http://127.0.0.1:5000").rstrip("/")


def _health() -> dict:
    r = requests.get(f"{_api_url()}/api/health", timeout=10)
    r.raise_for_status()
    return r.json()


def _upload(file_bytes: bytes, file_name: str) -> dict:
    files = {"file": (file_name, file_bytes)}
    r = requests.post(f"{_api_url()}/api/upload", files=files, timeout=180)
    r.raise_for_status()
    return r.json()


def _query(question: str, top_k: int, score_threshold: float) -> dict:
    payload = {
        "question": question,
        "top_k": top_k,
        "score_threshold": score_threshold,
    }
    r = requests.post(f"{_api_url()}/api/query", json=payload, timeout=180)
    r.raise_for_status()
    return r.json()


st.title("RAG Assistant")
st.caption("Upload a PDF/TXT and ask questions against your indexed documents.")

with st.sidebar:
    st.subheader("Connection")
    st.text_input("Flask API URL", key="api_base_url", value="http://127.0.0.1:5000")

    if st.button("Check API Health", use_container_width=True):
        try:
            health = _health()
            st.success(f"API OK | docs_in_store={health.get('docs_in_store', 0)}")
        except Exception as exc:
            st.error(f"Health check failed: {exc}")

st.subheader("1) Upload Document")
uploaded_file = st.file_uploader("Choose a PDF or TXT file", type=["pdf", "txt"])
force = st.checkbox("Force re-ingest duplicate file", value=False)

if st.button("Upload", type="primary", use_container_width=True):
    if not uploaded_file:
        st.warning("Please choose a file first.")
    else:
        try:
            endpoint = f"{_api_url()}/api/upload"
            if force:
                endpoint = f"{endpoint}?force=true"
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            response = requests.post(endpoint, files=files, timeout=180)
            data = response.json()
            if response.ok:
                st.success("Upload processed successfully.")
                st.json(data)
            else:
                st.error(data.get("error", "Upload failed."))
        except Exception as exc:
            st.error(f"Upload failed: {exc}")

st.subheader("2) Ask Question")
question = st.text_area("Your question", placeholder="What is AGR and how is it calculated?")
col1, col2 = st.columns(2)
with col1:
    top_k = st.slider("top_k", min_value=1, max_value=10, value=5)
with col2:
    score_threshold = st.slider("score_threshold", min_value=0.0, max_value=1.0, value=0.25)

if st.button("Run Query", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        try:
            result = _query(question.strip(), top_k=top_k, score_threshold=score_threshold)
            st.markdown("### Answer")
            st.write(result.get("answer", ""))
            st.markdown("### Confidence")
            st.write(result.get("confidence", 0.0))
            st.markdown("### Sources")
            sources = result.get("sources", [])
            if sources:
                st.dataframe(sources, use_container_width=True)
            else:
                st.info("No sources returned.")
            with st.expander("Raw Response"):
                st.code(json.dumps(result, indent=2), language="json")
        except requests.HTTPError as exc:
            try:
                err_data = exc.response.json()
                st.error(err_data.get("error", str(exc)))
            except Exception:
                st.error(str(exc))
        except Exception as exc:
            st.error(f"Query failed: {exc}")

st.divider()
st.caption(f"Workspace: {Path.cwd()}")

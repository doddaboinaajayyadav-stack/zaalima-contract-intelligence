import json
import requests
import streamlit as st

st.set_page_config(page_title="Contract Intelligence", layout="wide")
st.title("AI-Powered Contract Intelligence")
st.caption("PDF/DOCX → extraction/OCR → clause classification → entities → risk → semantic search")

api = st.sidebar.text_input("API URL", "http://127.0.0.1:8000")
file = st.file_uploader("Upload a contract", type=["pdf", "docx", "txt", "md"])

if file and st.button("Analyze Contract"):
    with st.spinner("Analyzing contract..."):
        response = requests.post(f"{api}/analyze", files={"file": (file.name, file.getvalue())}, timeout=180)
    if response.ok:
        data = response.json()
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk Score", data["risk"]["score"])
        c2.metric("Verdict", data["risk"]["verdict"])
        c3.metric("Chunks", data["chunks"])
        st.subheader("Detected Clauses")
        st.write(data["classification"]["present"] or "No high-confidence target clauses detected")
        st.subheader("Entities")
        st.json(data["entities"])
        st.subheader("Risk Contributions")
        st.json(data["risk"]["risk_contributions"])
        st.subheader("Full JSON")
        st.code(json.dumps(data, indent=2), language="json")
    else:
        st.error(response.text)

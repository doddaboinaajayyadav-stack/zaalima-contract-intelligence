import json
import requests
import streamlit as st

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Contract Intelligence",
    page_icon="📄",
    layout="wide"
)

# ---------------------------------------------------------
# Custom Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #6b7280;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .risk-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        background-color: #f8fafc;
        text-align: center;
    }

    .risk-score {
        font-size: 32px;
        font-weight: 700;
    }

    .verdict {
        font-size: 28px;
        font-weight: 700;
    }

    .section-title {
        font-size: 25px;
        font-weight: 650;
        margin-top: 25px;
    }

    .clause-box {
        padding: 12px 16px;
        margin: 7px 0;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        background-color: #f8fafc;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">📄 AI-Powered Contract Intelligence</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'PDF/DOCX → extraction/OCR → clause classification → '
    'entity extraction → risk analysis → semantic search'
    '</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    api = st.text_input(
        "API URL",
        "http://127.0.0.1:8001"
    )

    st.divider()

    st.info(
        "Upload a contract and click "
        "**Analyze Contract** to generate an AI-powered risk report."
    )

# ---------------------------------------------------------
# Upload Section
# ---------------------------------------------------------
st.subheader("📤 Upload a Contract")

file = st.file_uploader(
    "Choose a contract document",
    type=["pdf", "docx", "txt", "md"],
    help="Supported formats: PDF, DOCX, TXT and Markdown"
)

if file:
    st.success(f"Uploaded: **{file.name}**")

    if st.button(
        "🔍 Analyze Contract",
        type="primary",
        use_container_width=False
    ):

        with st.spinner("Analyzing contract..."):

            try:
                response = requests.post(
                    f"{api}/analyze",
                    files={
                        "file": (
                            file.name,
                            file.getvalue()
                        )
                    },
                    timeout=180
                )

            except requests.exceptions.RequestException as e:
                st.error(
                    f"Could not connect to the API.\n\n"
                    f"Make sure the FastAPI server is running.\n\n"
                    f"Error: {e}"
                )
                st.stop()

        # -------------------------------------------------
        # API Response
        # -------------------------------------------------
        if response.ok:

            data = response.json()

            risk = data["risk"]
            score = risk["score"]
            verdict = risk["verdict"]
            chunks = data["chunks"]

            # -------------------------------------------------
            # Summary Metrics
            # -------------------------------------------------
            st.markdown(
                '<div class="section-title">📊 Contract Risk Summary</div>',
                unsafe_allow_html=True
            )

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Risk Score",
                    f"{score:.2f}"
                )

            with c2:
                st.metric(
                    "Risk Verdict",
                    verdict
                )

            with c3:
                st.metric(
                    "Text Chunks",
                    chunks
                )

            # -------------------------------------------------
            # Verdict Message
            # -------------------------------------------------
            if verdict == "CRITICAL":
                st.error(
                    "🚨 CRITICAL RISK — This contract requires "
                    "careful review."
                )

            elif verdict == "HIGH":
                st.warning(
                    "⚠️ HIGH RISK — Several potentially risky "
                    "contract provisions were detected."
                )

            elif verdict == "MEDIUM":
                st.warning(
                    "🟡 MEDIUM RISK — Some provisions may require "
                    "additional review."
                )

            else:
                st.success(
                    "🟢 LOW RISK — No major high-risk provisions "
                    "were detected."
                )

            # -------------------------------------------------
            # Detected Clauses
            # -------------------------------------------------
            st.markdown(
                '<div class="section-title">📑 Detected Clauses</div>',
                unsafe_allow_html=True
            )

            clauses = data["classification"]["present"]

            if clauses:

                for clause in clauses:
                    st.markdown(
                        f'<div class="clause-box">🔹 {clause}</div>',
                        unsafe_allow_html=True
                    )

            else:
                st.info(
                    "No high-confidence target clauses detected."
                )

            # -------------------------------------------------
            # Entities
            # -------------------------------------------------
            st.markdown(
                '<div class="section-title">🏢 Extracted Entities</div>',
                unsafe_allow_html=True
            )

            entities = data["entities"]

            e1, e2, e3, e4 = st.columns(4)

            with e1:
                st.metric(
                    "Organizations",
                    len(entities.get("organizations", []))
                )

            with e2:
                st.metric(
                    "Dates",
                    len(entities.get("dates", []))
                )

            with e3:
                st.metric(
                    "Monetary Values",
                    len(entities.get("monetary_values", []))
                )

            with e4:
                st.metric(
                    "Jurisdictions",
                    len(entities.get("jurisdictions", []))
                )

            with st.expander("View Extracted Entities"):
                st.json(entities)

            # -------------------------------------------------
            # Risk Contributions
            # -------------------------------------------------
            st.markdown(
                '<div class="section-title">⚠️ Risk Contributions</div>',
                unsafe_allow_html=True
            )

            contributions = risk["risk_contributions"]

            if contributions:

                # Convert dictionary into chart-friendly data
                chart_data = {
                    "Clause": list(contributions.keys()),
                    "Risk Contribution": list(contributions.values())
                }

                st.bar_chart(
                    chart_data,
                    x="Clause",
                    y="Risk Contribution"
                )

                with st.expander("View Risk Contribution Details"):
                    for clause, value in contributions.items():
                        st.write(
                            f"**{clause}**: {value:.2f}"
                        )

            # -------------------------------------------------
            # Contract Information
            # -------------------------------------------------
            st.markdown(
                '<div class="section-title">📄 Contract Information</div>',
                unsafe_allow_html=True
            )

            info1, info2 = st.columns(2)

            with info1:
                st.write(
                    f"**Filename:** {data.get('filename', file.name)}"
                )

                st.write(
                    f"**Text Characters:** "
                    f"{data.get('text_characters', 'N/A')}"
                )

            with info2:
                st.write(
                    f"**Chunks:** {data.get('chunks', 'N/A')}"
                )

                st.write(
                    f"**Vector Indexed:** "
                    f"{'Yes' if data.get('vector_indexed') else 'No'}"
                )

            # -------------------------------------------------
            # Full JSON
            # -------------------------------------------------
            st.markdown(
                '<div class="section-title">🧾 Full Analysis JSON</div>',
                unsafe_allow_html=True
            )

            st.json(data)

        else:

            st.error(
                f"API Error ({response.status_code})"
            )

            st.code(
                response.text
            )

else:
    st.info(
        "👆 Upload a PDF, DOCX, TXT or MD contract to begin analysis."
    )
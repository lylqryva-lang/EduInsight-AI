
from pathlib import Path
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

PROJECT = Path(__file__).resolve().parent
RESULTS = PROJECT / "results"
RAW_FOLDER = PROJECT / "data" / "raw"

PREDICTION_PATH = RESULTS / "model_evaluation" / "integrated_predictions.csv"
TOPIC_PATH = RESULTS / "topic_analysis" / "feedback_with_topics.csv"

st.set_page_config(
    page_title="EduInsight AI",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 EduInsight AI")
st.subheader("AI-Powered Education Feedback Analysis")

st.markdown("""
EduInsight AI analyzes educational feedback using
Natural Language Processing and Machine Learning.
It helps identify student opinions and educational insights.
""")

st.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Dataset",
        "Topic Analysis",
        "Sentiment Analysis",
        "Feedback Prediction",
        "About Project"
    ]
)

@st.cache_data
def load_predictions():
    if PREDICTION_PATH.exists():
        return pd.read_csv(PREDICTION_PATH)
    return pd.DataFrame()

if page == "Dashboard":
    st.header("Project Overview")

    df = load_predictions()

    col1, col2, col3 = st.columns(3)
    col1.metric("Platform", "EduInsight AI")
    col2.metric("Technology", "NLP & ML")
    col3.metric("Total Reviews", len(df) if not df.empty else 0)

    st.info("Education Feedback Analytics Platform")

    st.markdown("### Main Features")
    st.write("- Sentiment classification")
    st.write("- Aspect-based feedback analysis")
    st.write("- Category prediction")
    st.write("- Dataset exploration")

elif page == "Dataset":
    st.header("📁 EduRABSA Dataset")

    if RAW_FOLDER.exists():
        files = []

        for pattern in ["*.json", "*.txt", "*.csv"]:
            files.extend(RAW_FOLDER.rglob(pattern))

        st.metric("Total Dataset Files", len(files))

        if files:
            file_info = pd.DataFrame([
                {
                    "File Name": f.name,
                    "Format": f.suffix,
                    "Size (KB)": round(f.stat().st_size / 1024, 2)
                }
                for f in files
            ])

            st.dataframe(file_info, use_container_width=True)

            selected = st.selectbox(
                "Select a file",
                files,
                format_func=lambda f: f.name
            )

            try:
                if selected.suffix == ".json":
                    with open(selected, "r", encoding="utf-8-sig") as f:
                        content = json.load(f)
                    st.json(content)

                elif selected.suffix == ".csv":
                    st.dataframe(
                        pd.read_csv(selected).head(50),
                        use_container_width=True
                    )

                elif selected.suffix == ".txt":
                    with open(selected, "r", encoding="utf-8-sig") as f:
                        st.text(f.read(5000))

            except Exception as e:
                st.error(f"File reading error: {e}")
    else:
        st.warning("Raw dataset folder is not available.")

elif page == "Topic Analysis":
    st.header("📊 Topic Analysis")

    if TOPIC_PATH.exists():
        df = pd.read_csv(TOPIC_PATH)
        st.metric("Total Reviews", len(df))
        st.dataframe(df.head(100), use_container_width=True)

        if "dominant_topic" in df.columns:
            st.bar_chart(df["dominant_topic"].value_counts())
    else:
        st.info("Topic analysis results are not available.")

elif page == "Sentiment Analysis":
    st.header("💬 Sentiment Analysis")

    df = load_predictions()

    if not df.empty and "predicted_sentiment" in df.columns:
        st.metric("Total Reviews", len(df))

        counts = df["predicted_sentiment"].value_counts()
        st.subheader("Sentiment Distribution")
        st.bar_chart(counts)

        st.subheader("Prediction Results")
        st.dataframe(df.head(100), use_container_width=True)
    else:
        st.warning("Prediction results are not available.")

elif page == "Feedback Prediction":
    st.header("🔍 Feedback Prediction")

    st.info(
        "The trained models are stored locally. "
        "Individual feedback prediction requires all model files."
    )

    st.warning(
        "This feature requires the trained model files, "
        "including aspect_model.pkl."
    )

elif page == "About Project":
    st.header("About EduInsight AI")

    st.markdown("""
    **EduInsight AI** is an AI-powered educational feedback
    analysis platform.

    **Technologies Used:**
    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - NLP
    - TF-IDF
    - Matplotlib
    - Streamlit

    **Dataset:** EduRABSA

    **Goal:** To help educational institutions understand
    student feedback and make data-driven decisions.
    """)

st.divider()
st.caption("EduInsight AI | AI-Powered Education Analytics")

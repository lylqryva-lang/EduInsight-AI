
from pathlib import Path
import json
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =====================================
# 1. PROJECT CONFIGURATION
# =====================================

PROJECT = Path(__file__).resolve().parent
RESULTS = PROJECT / "results"
RAW_FOLDER = PROJECT / "data" / "raw"

PREDICTION_PATH = (
    RESULTS / "model_evaluation" / "integrated_predictions.csv"
)
st.write("CSV path:", str(PREDICTION_PATH))
st.write("File exists:", PREDICTION_PATH.exists())
TOPIC_PATH = (
    RESULTS / "topic_analysis" / "feedback_with_topics.csv"
)

# =====================================
# 2. PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="EduInsight AI",
    page_icon="🎓",
    layout="wide"
)

# =====================================
# 3. HEADER
# =====================================

st.title("🎓 EduInsight AI")
st.subheader("AI-Powered Education Feedback Analysis")

st.markdown("""
EduInsight AI is an intelligent platform designed to analyze
student feedback using Natural Language Processing and Machine Learning.

It helps educational institutions understand student opinions,
identify common topics and improve educational quality.
""")

st.divider()

# =====================================
# 4. DATA LOADING
# =====================================

@st.cache_data
def load_predictions():
    if PREDICTION_PATH.exists():
        return pd.read_csv(PREDICTION_PATH)
    return pd.DataFrame()


# =====================================
# 5. SIDEBAR NAVIGATION
# =====================================

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

# =====================================
# 6. DASHBOARD
# =====================================

if page == "Dashboard":

    st.header("Project Overview")

    df = load_predictions()

    col1, col2, col3 = st.columns(3)

    col1.metric("Platform", "EduInsight AI")
    col2.metric("Technology", "NLP & ML")
    col3.metric(
        "Total Reviews",
        len(df) if not df.empty else 0
    )

    st.info(
        "Welcome to EduInsight AI – "
        "your Education Feedback Analytics Platform."
    )

    st.markdown("### Main Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Sentiment Analysis**
        - Positive feedback
        - Negative feedback
        - Neutral feedback
        """)

    with col2:
        st.markdown("""
        **Aspect-Based Analysis**
        - Aspect identification
        - Category classification
        - Educational insights
        """)

# =====================================
# 7. DATASET EXPLORER
# =====================================

elif page == "Dataset":

    st.header("📁 EduRABSA Dataset")

    if RAW_FOLDER.exists():

        data_files = []

        for pattern in ["*.json", "*.txt", "*.csv"]:
            data_files.extend(RAW_FOLDER.rglob(pattern))

        st.metric("Total Dataset Files", len(data_files))

        if data_files:

            file_info = []

            for file in data_files:
                file_info.append({
                    "File Name": file.name,
                    "Format": file.suffix,
                    "Size (KB)": round(
                        file.stat().st_size / 1024, 2
                    )
                })

            st.dataframe(
                pd.DataFrame(file_info),
                use_container_width=True
            )

            selected_file = st.selectbox(
                "Select a file to inspect",
                data_files,
                format_func=lambda x: x.name
            )

            try:

                if selected_file.suffix == ".json":

                    with open(
                        selected_file,
                        "r",
                        encoding="utf-8-sig"
                    ) as f:
                        content = json.load(f)

                    st.subheader("JSON Structure")
                    st.write("Data type:", type(content).__name__)

                    if isinstance(content, dict):
                        st.write("Keys:", list(content.keys()))
                        st.json(content)

                    elif isinstance(content, list):
                        st.write("Number of elements:", len(content))
                        st.json(content[:10])

                    else:
                        st.write(content)

                elif selected_file.suffix == ".txt":

                    with open(
                        selected_file,
                        "r",
                        encoding="utf-8-sig"
                    ) as f:
                        st.text(f.read(5000))

                elif selected_file.suffix == ".csv":

                    df = pd.read_csv(selected_file)

                    st.metric("Records", len(df))
                    st.metric("Columns", len(df.columns))

                    st.dataframe(
                        df.head(20),
                        use_container_width=True
                    )

            except Exception as e:
                st.error(f"File reading error: {e}")

        else:
            st.warning("No supported dataset files found.")

    else:
        st.warning(
            "Raw dataset folder is not available in the repository."
        )

# =====================================
# 8. TOPIC ANALYSIS
# =====================================

elif page == "Topic Analysis":

    st.header("📊 Topic Analysis")

    if TOPIC_PATH.exists():

        df = pd.read_csv(TOPIC_PATH)

        st.metric("Total Reviews", len(df))

        st.dataframe(
            df.head(100),
            use_container_width=True
        )

        if "dominant_topic" in df.columns:

            counts = df["dominant_topic"].value_counts()

            fig, ax = plt.subplots()

            counts.plot(kind="bar", ax=ax)

            ax.set_title("Topic Distribution")
            ax.set_xlabel("Topic")
            ax.set_ylabel("Number of Reviews")

            st.pyplot(fig)
            plt.close(fig)

    else:
        st.info("Topic analysis results are not available.")

# =====================================
# 9. SENTIMENT ANALYSIS
# =====================================

elif page == "Sentiment Analysis":

    st.header("💬 Sentiment Analysis")

    df = load_predictions()

    if not df.empty and "predicted_sentiment" in df.columns:

        st.metric("Total Reviews", len(df))

        st.subheader("Sentiment Distribution")

        sentiment_counts = df["predicted_sentiment"].value_counts()

        st.bar_chart(sentiment_counts)

        st.subheader("Prediction Results")

        st.dataframe(
            df.head(100),
            use_container_width=True
        )

    else:
        st.warning(
            "Prediction results are not available. "
            "Check the CSV file path and repository contents."
        )

# =====================================
# 10. FEEDBACK PREDICTION
# =====================================

elif page == "Feedback Prediction":

    st.header("🔍 Feedback Prediction")

    st.info(
        "Individual feedback prediction requires the trained "
        "machine learning models."
    )

    st.warning(
        "The trained model files must be available "
        "in the models/absa directory."
    )

# =====================================
# 11. ABOUT PROJECT
# =====================================

elif page == "About Project":

    st.header("About EduInsight AI")

    st.markdown("""
    EduInsight AI is an AI-powered educational feedback
    analysis platform.

    **Technologies Used:**
    - Python
    - Pandas
    - NumPy
    - Scikit-learn
    - Natural Language Processing
    - TF-IDF
    - Matplotlib
    - Streamlit

    **Dataset:** EduRABSA

    **Main Goal:**

    To help educational institutions analyze student feedback
    and make data-driven decisions.
    """)

# =====================================
# 12. FOOTER
# =====================================

st.divider()

st.caption(
    "EduInsight AI | AI-Powered Education Analytics"
)

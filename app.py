
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
raw_folder = PROJECT / "data" / "raw"

topic_path = RESULTS / "topic_analysis" / "feedback_with_topics.csv"
prediction_path = RESULTS / "model_evaluation" / "model_predictions.csv"

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
# 4. SIDEBAR NAVIGATION
# =====================================

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Dataset",
        "Topic Analysis",
        "Sentiment Analysis",
        "About Project"
    ]
)

# =====================================
# 5. DASHBOARD
# =====================================

if page == "Dashboard":

    st.header("Project Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Platform", "EduInsight AI")
    col2.metric("Technology", "NLP & ML")
    col3.metric("Project Status", "Development")

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
        **Topic Analysis**

        - Feedback classification
        - Common topic identification
        - Educational insights
        """)

# =====================================
# 6. DATASET EXPLORER
# =====================================

elif page == "Dataset":

    st.header("📁 EduRABSA Dataset")
    st.write("Explore the dataset files and their structure.")

    if raw_folder.exists():

        data_files = []

        for pattern in ["*.json", "*.txt", "*.csv"]:
            data_files.extend(raw_folder.rglob(pattern))

        st.metric("Total Dataset Files", len(data_files))

        if data_files:

            file_info = []

            for file in data_files:
                file_info.append({
                    "File Name": file.name,
                    "Format": file.suffix,
                    "Size (KB)": round(
                        file.stat().st_size / 1024, 2
                    ),
                    "Folder": str(
                        file.parent.relative_to(raw_folder)
                    )
                })

            st.subheader("Available Files")

            st.dataframe(
                pd.DataFrame(file_info),
                use_container_width=True
            )

            selected_file = st.selectbox(
                "Select a file to inspect",
                data_files,
                format_func=lambda x: x.name
            )

            st.subheader("File Information")

            st.write("Name:", selected_file.name)
            st.write("Format:", selected_file.suffix)
            st.write(
                "Size:",
                round(selected_file.stat().st_size / 1024, 2),
                "KB"
            )

            try:

                if selected_file.suffix == ".json":

                    st.subheader("JSON Structure")

                    with open(
                        selected_file,
                        "r",
                        encoding="utf-8-sig"
                    ) as f:
                        content = json.load(f)

                    st.write(
                        "Data type:",
                        type(content).__name__
                    )

                    if isinstance(content, dict):

                        st.write("Keys:", list(content.keys()))

                        st.json(content)

                    elif isinstance(content, list):

                        st.write(
                            "Number of elements:",
                            len(content)
                        )

                        st.json(content[:10])

                    else:
                        st.write(content)

                elif selected_file.suffix == ".txt":

                    st.subheader("Text Preview")

                    with open(
                        selected_file,
                        "r",
                        encoding="utf-8-sig"
                    ) as f:
                        content = f.read(5000)

                    st.text(content)

                elif selected_file.suffix == ".csv":

                    st.subheader("CSV Preview")

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

        st.error("Raw dataset folder not found.")

# =====================================
# 7. TOPIC ANALYSIS
# =====================================

elif page == "Topic Analysis":

    st.header("📊 Topic Analysis")

    if topic_path.exists():

        df = pd.read_csv(topic_path)

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

        st.warning(
            "Topic analysis results are not available yet."
        )

# =====================================
# 8. SENTIMENT ANALYSIS
# =====================================

elif page == "Sentiment Analysis":

    st.header("💬 Sentiment Analysis")

    if prediction_path.exists():

        df = pd.read_csv(prediction_path)

        st.dataframe(
            df.head(100),
            use_container_width=True
        )

        if "predicted_sentiment" in df.columns:

            st.subheader("Sentiment Distribution")

            st.bar_chart(
                df["predicted_sentiment"].value_counts()
            )

    else:

        st.warning(
            "Sentiment model predictions are not available yet."
        )

# =====================================
# 9. ABOUT PROJECT
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

    **Main Goal:**

    To help educational institutions analyze student feedback
    and make data-driven decisions.
    """)

# =====================================
# 10. FOOTER
# =====================================

st.divider()

st.caption(
    "EduInsight AI | AI-Powered Education Analytics"
)

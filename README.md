# 🎓 EduInsight AI
### AI-Powered Education Feedback Analysis Platform

EduInsight AI is a machine learning-based platform designed to analyze educational feedback using Natural Language Processing (NLP) and Aspect-Based Sentiment Analysis (ABSA).

The platform helps identify student opinions, evaluate educational aspects, and extract meaningful insights from feedback.

## 🚀 Features

- Sentiment Classification (Positive, Negative, Neutral)
- Aspect Prediction
- Category Prediction
- Real-time Feedback Analysis
- Interactive Streamlit Dashboard
- Dataset Exploration and Visualization
- Model Evaluation

## 🧠 Machine Learning

The project uses:

- TF-IDF for text feature extraction
- Logistic Regression for classification
- Scikit-learn for model development and evaluation
- Aspect-Based Sentiment Analysis (ABSA)

### Dataset

The project uses the EduRABSA dataset.

Dataset source: https://zenodo.org/records/18500517

License: CC BY-NC-SA 4.0

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Jupyter Notebook

## 📊 Initial Model Results

| Metric | Result |
|---|---:|
| Sentiment Accuracy | 66.49% |
| Weighted F1-score | 0.69 |

The Neutral sentiment class requires further improvement.

## 💻 Installation

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📁 Project Structure

```text
EduInsight-AI/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── raw/
│
├── models/
│   └── absa/
│
├── notebooks/
│
└── results/
    ├── absa/
    └── model_evaluation/
```

## 🔮 Future Improvements

- Improve classification performance
- Enhance aspect extraction
- Add multilingual feedback support
- Develop advanced analytics
- Improve dashboard visualizations

## 👩‍💻 Author

Developed as an Artificial Intelligence and Machine Learning project.

**EduInsight AI — Turning Educational Feedback into Insights.**

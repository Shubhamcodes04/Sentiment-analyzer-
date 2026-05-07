# 🎭 Sentiment Dashboard
### Dual-model NLP sentiment analyzer — VADER + TextBlob

> Paste any text. Get instant sentiment analysis with rich visualizations.

No API key needed. Runs entirely on free, open-source NLP libraries.

---

## ✨ Features

- 🔵 **VADER** — Rule-based NLP, great for social media & informal text
- 🟣 **TextBlob** — Statistical NLP, great for formal writing
- 📊 **Model comparison chart** — See where both models agree or differ
- 🎯 **Subjectivity gauge** — How opinionated vs factual is the text?
- 🔍 **Sentence-by-sentence breakdown** — Color-coded per sentence
- ⬇️ **Downloadable reports** — Save analysis as Markdown

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| NLP Model 1 | VADER (vaderSentiment) |
| NLP Model 2 | TextBlob |
| Charts | Plotly |
| Language | Python 3.10+ |
| Deployment | Streamlit Cloud |

---

## 🚀 Getting Started

```bash
git clone https://github.com/YOUR_USERNAME/sentiment-dashboard.git
cd sentiment-dashboard

pip install -r requirements.txt
python -m textblob.download_corpora   # one-time download

streamlit run app.py
```

---

## 📁 Project Structure

```
sentiment-dashboard/
├── app.py           # Streamlit UI + Plotly charts
├── analyzer.py      # VADER + TextBlob NLP logic
├── requirements.txt
└── README.md
```

---

## 👨‍💻 Author

**Your Name** — B.Tech CSE @ JUET
[LinkedIn](https://linkedin.com/in/yourprofile) · [GitHub](https://github.com/yourusername)

---

*Built as a portfolio project for summer internship applications.*

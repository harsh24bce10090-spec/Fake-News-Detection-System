# 📰 Fake News Detector

A machine learning web application that detects fake news articles using Natural Language Processing (NLP) and Logistic Regression.

## 🚀 Features

- Real-time news classification (Real vs Fake)
- Confidence score for each prediction
- Clean, responsive web interface
- TF-IDF feature extraction
- Preprocessing (stemming, stopword removal)

## 🛠️ Tech Stack

- Python 3.8+
- Flask (Web Framework)
- Scikit-learn (ML Model)
- NLTK (Text Processing)
- HTML/CSS/JavaScript

## 📋 Prerequisites

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"

fake-news-detector/
├── app.py              # Flask web app
├── train_model.py      # Training script
├── predict.py          # Prediction script
├── requirements.txt    # Dependencies
├── models/            # Saved model files
├── templates/         # HTML templates
└── static/            # CSS files

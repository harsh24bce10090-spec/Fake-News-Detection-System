"""
Fake News Detector - Prediction Script
Use this to classify new text
"""

import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load the saved model and vectorizer
model = joblib.load('models/fake_news_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')

# Initialize preprocessing tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Clean and preprocess input text
    """
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

def predict_news(text):
    """
    Predict if news is Real (0) or Fake (1)
    Returns: prediction, confidence
    """
    # Preprocess
    cleaned = preprocess_text(text)
    
    # Vectorize
    features = vectorizer.transform([cleaned]).toarray()
    
    # Predict
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features)[0]
    
    result = "FAKE" if prediction == 1 else "REAL"
    confidence_score = confidence[prediction] * 100
    
    return result, confidence_score

def get_feedback(text, predicted_label, actual_label):
    """
    Collect user feedback for model improvement
    """
    if predicted_label == actual_label:
        return "Correct prediction"
    else:
        return "Incorrect prediction - would be used for retraining"

# Example usage
if __name__ == "__main__":
    # Test examples
    test_articles = [
        "Breaking news scientists discover alien life in secret government lab",
        "The Federal Reserve announced interest rates will remain unchanged",
    ]
    
    print("=" * 50)
    print("FAKE NEWS DETECTOR - TEST PREDICTIONS")
    print("=" * 50)
    
    for article in test_articles:
        result, confidence = predict_news(article)
        print(f"\n📰 Article: {article[:50]}...")
        print(f"🎯 Prediction: {result}")
        print(f"📊 Confidence: {confidence:.2f}%")

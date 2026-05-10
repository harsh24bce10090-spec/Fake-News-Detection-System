"""
Fake News Detector - Web Application
Run with: python app.py
Then open browser at http://127.0.0.1:5000
"""

from flask import Flask, render_template, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Initialize Flask app
app = Flask(__name__)

# Download NLTK data (first time only)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

# Load model and vectorizer
try:
    model = joblib.load('models/fake_news_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    print("✅ Model loaded successfully")
except FileNotFoundError:
    print("❌ Model not found. Run train_model.py first")
    model = None
    vectorizer = None

# Initialize preprocessing tools
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """Clean and preprocess input text"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

def predict_news(text):
    """Predict if news is real or fake"""
    if model is None or vectorizer is None:
        return "ERROR", 0.0
    
    cleaned = preprocess_text(text)
    features = vectorizer.transform([cleaned]).toarray()
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features)[0][prediction] * 100
    
    result = "Fake News" if prediction == 1 else "Real News"
    return result, confidence

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text.strip():
            return jsonify({'error': 'Please enter some text'}), 400
        
        result, confidence = predict_news(text)
        
        return jsonify({
            'prediction': result,
            'confidence': round(confidence, 2),
            'text_length': len(text.split())
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({'status': 'running', 'model_loaded': model is not None})

if __name__ == '__main__':
    app.run(debug=True)

"""
Fake News Detector - Model Training Script
Run this file first to train and save the model
"""

import pandas as pd
import numpy as np
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Download required NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Initialize stemmer and stopwords
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Clean and preprocess the input text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    
    # Tokenize and remove stopwords
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    
    # Join back to string
    return ' '.join(words)

def load_data():
    """
    Load dataset from CSV file
    Note: Download dataset from Kaggle 'fake-news' dataset
    """
    # Option 1: Load from CSV
    # df = pd.read_csv('data/news.csv')
    
    # Option 2: Create sample data for demonstration
    # In production, replace with actual dataset
    data = {
        'text': [
            "Breaking news shocking discovery scientists cannot explain",
            "Government hides truth from public about health crisis",
            "You won't believe what this celebrity did yesterday",
            "Miracle cure that doctors don't want you to know about",
            "The economy is showing steady growth this quarter",
            "President signs new bill into law today",
            "Scientists publish research on climate change effects",
            "Company announces quarterly earnings beat expectations",
            "Shocking video exposes truth about election fraud",
            "New study confirms benefits of regular exercise"
        ],
        'label': [1, 1, 1, 1, 0, 0, 0, 0, 1, 0]
    }
    # 1 = Fake, 0 = Real
    
    df = pd.DataFrame(data)
    print(f"Dataset loaded with {len(df)} samples")
    print(f"Class distribution:\n{df['label'].value_counts()}")
    
    return df

def main():
    print("=" * 50)
    print("FAKE NEWS DETECTOR - TRAINING PIPELINE")
    print("=" * 50)
    
    # 1. Load data
    print("\n[1/6] Loading dataset...")
    df = load_data()
    
    # 2. Preprocess text
    print("\n[2/6] Preprocessing text...")
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    
    # 3. Feature extraction using TF-IDF
    print("\n[3/6] Extracting TF-IDF features...")
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(df['cleaned_text']).toarray()
    y = df['label'].values
    
    print(f"Feature matrix shape: {X.shape}")
    
    # 4. Split data
    print("\n[4/6] Splitting train/test data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 5. Train model
    print("\n[5/6] Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # 6. Evaluate
    print("\n[6/6] Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Model trained successfully!")
    print(f"📊 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real', 'Fake']))
    
    # Save model and vectorizer
    print("\n💾 Saving model and vectorizer...")
    joblib.dump(model, 'models/fake_news_model.pkl')
    joblib.dump(tfidf, 'models/tfidf_vectorizer.pkl')
    
    print("\n✅ Model saved to 'models/fake_news_model.pkl'")
    print("✅ Vectorizer saved to 'models/tfidf_vectorizer.pkl'")
    
    return model, tfidf

if __name__ == "__main__":
    main()

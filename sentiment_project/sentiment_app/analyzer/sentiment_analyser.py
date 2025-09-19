import numpy as np
import pandas as pd
import re
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

# ============== DATA PREPROCESSING ==============

class TweetPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_tweet(self, text):
        """Clean and preprocess tweet text"""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove user mentions but keep hashtags as they might be meaningful
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtag symbol but keep the word
        text = re.sub(r'#', '', text)
        
        # Remove special characters and digits but keep spaces
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # For gaming tweets, we might want to keep some gaming-specific terms
        # Don't remove all stopwords for better context in gaming tweets
        tokens = text.split()
        
        # Only remove very common stopwords that don't affect sentiment
        very_common_stops = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        tokens = [word for word in tokens if word not in very_common_stops or len(word) > 2]
        
        return ' '.join(tokens)

class SentimentPredictor:
    """Class for loading and using the saved model"""
    
    def __init__(self, model_name='twitter_sentiment'):
        # Load model
        self.model = load_model(f'{model_name}_model.keras')
        
        # Load tokenizer
        with open(f'{model_name}_tokenizer.pickle', 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        
        # Load config
        with open(f'{model_name}_config.pickle', 'rb') as handle:
            self.config = pickle.load(handle)
        
        # Initialize preprocessor
        self.preprocessor = TweetPreprocessor()
        
        print("✅ Model loaded and ready for predictions!")
    
    def predict_sentiment(self, text):
        """Predict sentiment for a single text"""
        # Clean text
        cleaned = self.preprocessor.clean_tweet(text)
        
        # Tokenize and pad
        sequence = self.tokenizer.texts_to_sequences([cleaned])
        padded = pad_sequences(sequence, maxlen=self.config['max_len'], 
                             padding='post', truncating='post')
        
        # Predict
        prediction = self.model.predict(padded, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        return {
            'text': text,
            'cleaned_text': cleaned,
            'sentiment': self.config['sentiment_labels'][predicted_class],
            'confidence': float(confidence),
            'probabilities': {
                label: float(prob) 
                for label, prob in zip(self.config['sentiment_labels'], prediction[0])
            }
        }
    
    def predict_batch(self, texts):
        """Predict sentiment for multiple texts"""
        results = []
        for text in texts:
            results.append(self.predict_sentiment(text))
        return results
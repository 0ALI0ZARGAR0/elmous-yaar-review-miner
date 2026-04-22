import pandas as pd
from hazm import Normalizer, WordTokenizer, stopwords_list
from transformers import pipeline

class SentimentAnalyzer:
    def __init__(self):
        # Tools for Persian Text Preprocessing
        self.normalizer = Normalizer()
        self.tokenizer = WordTokenizer()
        # Using stopwords_list() to avoid ImportError on your system
        self.stopwords = set(stopwords_list())
        
        # Bonus Section: Using ParsBERT (Transformer model)
        # Pre-trained on Persian sentiment datasets
        model_id = "HooshvareLab/bert-fa-base-uncased-sentiment-digikala"
        try:
            self.analyzer = pipeline("sentiment-analysis", model=model_id)
        except Exception as e:
            print(f"Error loading transformer model: {e}")
            self.analyzer = None

    def full_preprocess(self, text):
        """Normalization, Tokenization, and Stopword removal."""
        if not isinstance(text, str) or text.strip() == "":
            return ""
        
        # 1. Normalization
        normalized = self.normalizer.normalize(text)
        
        # 2. Tokenization
        tokens = self.tokenizer.tokenize(normalized)
        
        # 3. Stopword Removal
        cleaned_tokens = [t for t in tokens if t not in self.stopwords]
        
        return " ".join(cleaned_tokens)

    def get_sentiment(self, text):
        """Returns Label and Score for the given text."""
        preprocessed = self.full_preprocess(text)
        if not preprocessed.strip() or self.analyzer is None:
            return "neutral", 0.5
        
        # Inference (Limited to 512 tokens for BERT compatibility)
        result = self.analyzer(preprocessed[:512])[0]
        return result['label'], result['score']
    
    def get_top_keywords(self, text_list, n=20):
        """
        Extracts the most frequent meaningful words from a list of texts.
        1. Preprocesses each text (Normalize, Tokenize, Stopwords).
        2. Filters out short, non-informative words.
        3. Returns a list of (word, frequency) tuples.
        """
        from collections import Counter
        
        all_cleaned_words = []
        for text in text_list:
            # استفاده از متد پیش‌پردازش که قبلاً نوشتیم
            cleaned_text = self.full_preprocess(str(text))
            words = cleaned_text.split()
            # فیلتر کردن کلمات کمتر از ۳ حرف برای دقت بیشتر
            words = [w for w in words if len(w) > 2]
            all_cleaned_words.extend(words)
        
        return Counter(all_cleaned_words).most_common(n)
    

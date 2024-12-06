
""" Text Preprocessing """

from nltk.stem import WordNetLemmatizer
from nltk import download
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
import re

class TextPreprocessor(BaseEstimator, TransformerMixin):

    """ Modular text preprocessing class """

    # Emotion Dictionary
    EMOTICONS = {
        r':D': '[LAUGHING]',
        r';\)': '[WINK]',
        r';-\)': '[WINK]',
        r';D': '[LAUGHINGWINK]',
        r':o': '[SURPRISED]',
        r'=\(': '[SAD]',
        r':\[': '[SAD]',
        r':-\(': '[SAD]',
        r':\(': '[SAD]',
        r'=\/': '[CONFUSED]',
        r'\b83\b': '[LOVE]',
        r'8p': '[PLAYFUL]',
        r':P': '[PLAYFUL]',
        r':3': '[CUTE]',
        r':]': '[HAPPY]',
        r'=]': '[HAPPY]',
        r'=\)': '[HAPPY]',
        r'=D': '[LAUGHING]',
        r':-/': '[CONFUSED]',
        r':>': '[SMUG]',
        r'8/': '[CONFUSED]',
        r':<': '[DISPLEASED]',
        r':O': '[SURPRISED]',
        r'=o': '[SURPRISED]',
        r':-\|': '[NEUTRAL]',
        r':-\)': '[HAPPY]',
        r':\)': '[HAPPY]',
    }

    def __init__(self, 
                 use_lemmatization = True,
                 use_stopwords = True,
                 lowercase = True,
                 remove_punctuation = True,
                 process_emoticons = True
                ):
        self.use_lemmatization = use_lemmatization
        self.use_stopwords = use_stopwords
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        self.process_emoticons = process_emoticons

        if self.use_lemmatization:
            download('wordnet')
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.lemmatizer = None

        if self.use_stopwords:
            download('stopwords')
            self.stopwords = set(stopwords.words('english'))
        else:
            self.stopwords = None

    def fit(self, X, y=None):

        """ Fit method for the transformer, does nothing interesting """

        return self

    def transform(self, X):

        """ Transform a list of texts using the preprocessing steps """
        
        # Handle both single string and list/array of strings
        if isinstance(X, str):
            return self._preprocess(X)
        
        return [self._preprocess(text) for text in X]

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)
    
    def _process_emoticons(self, text):
    
        """ Replace emoticons with their corresponding emotion tags """
        
        for pattern, replacement in self.EMOTICONS.items():
            text = re.sub(pattern, f' {replacement} ', text)
        
        return text
    
    def _preprocess(self, text):

        """Preprocess a single document."""
        
        # Process emoticons before lowercase to preserve patterns
        if self.process_emoticons:
            text = self._process_emoticons(text)

        if self.lowercase:
            # Don't lowercase emotion tags
            emotion_tags = re.findall(r'\[([A-Z]+)\]', text)
            text = text.lower()
            
            # Restore uppercase emotion tags
            for tag in emotion_tags:
                text = text.replace(f'[{tag.lower()}]', f'[{tag}]')

        if self.remove_punctuation:
            # Save emotion tags
            emotion_tags = re.findall(r'\[[A-Z]+\]', text)
            
            # Remove punctuation
            text = re.sub(r'[^\w\s]', '', text)
            
            # Restore emotion tags
            for tag in emotion_tags:
                text = text.replace(tag.strip('[]'), tag)

        tokens = text.split()

        if self.use_stopwords:
            tokens = [word for word in tokens if word not in self.stopwords]

        if self.use_lemmatization:
            # Don't lemmatize emotion tags
            tokens = [
                word if word.startswith('[') and word.endswith(']')
                else self.lemmatizer.lemmatize(word)
                for word in tokens
            ]

        return ' '.join(tokens)

# Example with pandas DataFrame
# import pandas as pd

# Assuming your data looks like this:
# df = pd.DataFrame({
#     'text': ["I love this product :)", "Not happy with it :(", ...],
#     'label': [1, 0, ...]
# })

# 1. Basic Pipeline Setup
# def create_text_pipeline():
#     return Pipeline([
#         ('preprocessor', TextPreprocessor(
#             use_lemmatization=True,
#             use_stopwords=True,
#             lowercase=True,
#             remove_punctuation=True,
#             process_emoticons=True
#         )),
#         ('vectorizer', TfidfVectorizer()),
#         ('classifier', LinearSVC())
#     ])

# # 2. Train-Test Split
# X_train, X_test, y_train, y_test = train_test_split(
#     df['text'],
#     df['label'],
#     test_size=0.2,
#     random_state=42
# )

# # 3. Fit and Predict
# pipeline = create_text_pipeline()
# pipeline.fit(X_train, y_train)

# # Make predictions
# predictions = pipeline.predict(X_test)

# # 4. Evaluate
# from sklearn.metrics import classification_report
# print(classification_report(y_test, predictions))
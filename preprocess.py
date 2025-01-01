import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Download NLTK resources (if not already downloaded)
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')

# Define stopwords and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess_text(text):
    """Normalize and preprocess text."""
    # Handle non-string or NaN values
    if not isinstance(text, str):
        return ""
    
    # Lowercase the text
    text = text.lower()
    
    # Remove punctuation and numbers
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    
    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stopwords and apply stemming
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    
    # Rejoin the tokens into a single string
    return ' '.join(tokens)


# Load the processed dataset (if saved earlier)
df = pd.read_csv("processed_data.csv")

# Handle missing values
df['abstract'] = df['abstract'].fillna('')

# Apply preprocessing
df['title_cleaned'] = df['title'].apply(preprocess_text)
df['abstract_cleaned'] = df['abstract'].apply(preprocess_text)

# Save the cleaned dataset for future use
df.to_csv("cleaned_data.csv", index=False)

# Display a preview of the cleaned text
print(df[['title', 'title_cleaned', 'abstract', 'abstract_cleaned']].head())

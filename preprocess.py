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



## FEATURE EXTRACTION
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Combine title and abstract into one feature
df['combined_text'] = df['title_cleaned'] + " " + df['abstract_cleaned']

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Adjust max_features as needed

# Transform combined text
X = tfidf_vectorizer.fit_transform(df['combined_text'])

# Target variable: Journal names
y = df['journal']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save the vectorizer for future use
import joblib
joblib.dump(tfidf_vectorizer, "tfidf_vectorizer.pkl")



## MODEL TRAINING
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "journal_recommendation_model.pkl")

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

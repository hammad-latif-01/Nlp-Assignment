import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import string

# Download required NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# New sample corpus related to space exploration
texts = [
    "NASA launched the Artemis program to return humans to the Moon.",
    "Mars rover Perseverance is designed to seek signs of ancient life.",
    "The Hubble Space Telescope has provided invaluable data since its launch.",
    "Astrobiology studies the potential for life on other planets.",
    "SpaceX is revolutionizing space travel with reusable rockets."
]

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Tokenization
    tokens = nltk.word_tokenize(text)

    # Stemming
    stemmer = PorterStemmer()
    tokens_stemmed = [stemmer.stem(token) for token in tokens]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens_lemmatized = [lemmatizer.lemmatize(token) for token in tokens_stemmed]

    # Join tokens back to string
    cleaned_text = ' '.join(tokens_lemmatized)
    return cleaned_text

# Preprocess texts and store results
cleaned_texts = [preprocess_text(text) for text in texts]

# Display some examples of before and after
for i in range(len(texts)):  # Show all examples
    print(f"Original Text: {texts[i]}")
    print(f"Cleaned Text: {cleaned_texts[i]}\n")

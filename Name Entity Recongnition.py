import random
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk

# Step to download necessary NLTK resources
nltk.download('vader_lexicon')

# Step 1: Generate a large text corpus (simulated)
sample_sentences = [
    "The food was absolutely delicious and well presented.",
    "I didn't enjoy the service; it was slow and unhelpful.",
    "This restaurant has a great atmosphere for dining.",
    "The prices are a bit high, but the quality is worth it.",
    "I had a wonderful experience with the staff; they were friendly.",
    "The dessert was disappointing; it lacked flavor.",
    "Overall, I would recommend this place to my friends.",
    "The portion sizes are generous and satisfying.",
    "I will not be returning due to the poor hygiene standards.",
    "A hidden gem in the city; I will definitely come back!"
]

# Create a larger corpus by repeating the sample sentences
corpus = " ".join(sample_sentences * 20000)  # Making it larger

# Split the corpus into sentences for analysis
sentences = corpus.split('. ')  # Simple sentence splitting

# Step 2: Sentiment Analysis using VADER
sia = SentimentIntensityAnalyzer()
vader_results = []

for sentence in sentences:
    scores = sia.polarity_scores(sentence)
    vader_results.append((sentence, scores))

# Step 3: Sentiment Analysis using TextBlob
textblob_results = []

for sentence in sentences:
    blob = TextBlob(sentence)
    textblob_results.append((sentence, blob.sentiment))

# Display the results for the first few sentences
print("VADER Sentiment Analysis Results:")
for i in range(3):  # Display first 3 results
    print(f"Sentence: {vader_results[i][0]}")
    print(f"Scores: {vader_results[i][1]}\n")

print("TextBlob Sentiment Analysis Results:")
for i in range(3):  # Display first 3 results
    print(f"Sentence: {textblob_results[i][0]}")
    print(f"Polarity: {textblob_results[i][1].polarity}, Subjectivity: {textblob_results[i][1].subjectivity}\n")

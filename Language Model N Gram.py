import random
import numpy as np
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import nltk

# Step to download necessary NLTK resources
nltk.download('vader_lexicon')

# Step 1: Generate a large text corpus (simulated)
sample_sentences = [
    "The trip to the mountains was breathtaking and rejuvenating.",
    "I was disappointed with the hotel service; it was quite lacking.",
    "Exploring the vibrant local markets was an unforgettable experience.",
    "The beaches were beautiful, but too crowded for my liking.",
    "I loved the guided tour; the guide was knowledgeable and friendly.",
    "The food was fantastic, each dish was a delight to taste.",
    "I wouldn't recommend visiting during the rainy season.",
    "The scenic views from the top were worth the hike.",
    "Public transportation was efficient and easy to navigate.",
    "I can't wait to return and explore more of this wonderful place!"
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

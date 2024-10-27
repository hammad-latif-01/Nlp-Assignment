import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Download necessary NLTK data
nltk.download('punkt')

# Step 1: Generate a large text corpus (simulated)
sample_sentences = [
    "Traveling opens your mind and broadens your horizons.",
    "Exploring new cultures can be an enriching experience.",
    "The beauty of nature can be found in national parks.",
    "Cuisine varies greatly from region to region.",
    "Traveling alone can lead to self-discovery.",
    "Visiting historical sites connects us to the past.",
    "Adventure tourism offers thrilling experiences like hiking and rafting.",
    "Sustainable travel practices help protect the environment.",
    "Photography allows you to capture memories from your travels.",
    "Planning a trip requires research and preparation."
]

# Create a larger corpus by repeating the sample sentences
corpus = [sentence for sentence in sample_sentences for _ in range(2000)]  # Making it larger

# Step 2: TF-IDF Vectorization for summarization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

# Calculate the importance of each sentence based on TF-IDF scores
sentence_scores = np.sum(tfidf_matrix, axis=1).A1  # Flatten the matrix to 1D array

# Get the indices of the top 3 sentences
top_indices = sentence_scores.argsort()[-3:][::-1]

# Display the top sentences as a summary
print("Summary of the text:")
for index in top_indices:
    print(f"- {corpus[index]}")

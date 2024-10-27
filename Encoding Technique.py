import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Step 1: Generate a large text corpus (simulated)
sample_sentences = [
    "Regular exercise is vital for maintaining good health.",
    "A balanced diet contributes to overall well-being.",
    "Mental health is just as important as physical health.",
    "Hydration plays a crucial role in bodily functions.",
    "Getting enough sleep improves cognitive function and mood.",
    "Mindfulness and meditation can reduce stress.",
    "Preventive healthcare can lead to early detection of diseases.",
    "Health education empowers individuals to make informed choices.",
    "Vaccinations are essential for preventing infectious diseases.",
    "Chronic illnesses require ongoing management and support."
]

# Create a larger corpus by repeating the sample sentences
corpus = [" ".join(sample_sentences) for _ in range(20000)]  # Making it larger

# Step 2: TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)

# Step 3: Displaying TF-IDF for some words
feature_names = vectorizer.get_feature_names_out()

# Example: Show the TF-IDF scores for the first document
dense = tfidf_matrix.todense()
doc_array = dense[0].A1  # Convert the first document to a 1D array

# Display the top 5 words with their TF-IDF scores
top_n = 5
indices = np.argsort(doc_array)[-top_n:][::-1]
print("Top words in the first document with TF-IDF scores:")
for index in indices:
    print(f"Word: {feature_names[index]}, TF-IDF: {doc_array[index]}")

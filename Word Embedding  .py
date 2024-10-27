import random
from gensim.utils import simple_preprocess
from gensim.models import FastText

# Step 1: Simulating a text corpus with random sentences about technology
sample_sentences = [
    "Artificial intelligence is transforming various industries.",
    "Blockchain technology ensures secure and transparent transactions.",
    "The Internet of Things connects devices and enhances data collection.",
    "Augmented reality is changing how we interact with digital content.",
    "5G technology promises faster connectivity and improved mobile experiences.",
    "Cybersecurity is crucial for protecting sensitive information.",
    "Cloud computing provides scalable resources and services over the internet.",
    "Data science involves analyzing complex data to extract insights.",
    "Quantum computing has the potential to revolutionize problem-solving.",
    "Robotics is advancing rapidly with innovations in automation."
]

# Create a larger corpus by repeating the sample sentences
corpus = sample_sentences * 1000  # Making it larger

# Step 2: Tokenize the corpus
tokenized_corpus = [simple_preprocess(sentence) for sentence in corpus]

# Step 3: Train FastText model
model = FastText(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, sg=1)

# Step 4: Explore the word embeddings
# Get vector for a specific word
word_vector = model.wv['artificial']
print(f"Vector for 'artificial': {word_vector}")

# Find most similar words
similar_words = model.wv.most_similar('artificial', topn=5)
print("Most similar words to 'artificial':")
for word, similarity in similar_words:
    print(f"{word}: {similarity}")

# Step 5: Save the model (optional)
model.save("fasttext_tech_model.bin")

# Step 6: Load the model (optional)
# loaded_model = FastText.load("fasttext_tech_model.bin")

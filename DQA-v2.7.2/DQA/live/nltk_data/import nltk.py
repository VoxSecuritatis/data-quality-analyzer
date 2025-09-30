# nltk_setup.py
# One-time setup script to download required NLTK resources

import nltk

# Download core resources
nltk.download("punkt")
nltk.download("punkt_tab")   # Required in NLTK 3.8+
nltk.download("stopwords")

# --- Verification ---
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Test tokenization
sample_text = "Hello world, this is a test!"
print("Tokenized:", word_tokenize(sample_text))

# Test stopwords
print("Stopwords sample:", stopwords.words("english")[:10])
import nltk

# Download all punkt related resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('tokenizers/punkt/english.pickle')

# Download stopwords
nltk.download('stopwords')

# Test tokenization
text = "This is a test sentence."
try:
    tokens = nltk.word_tokenize(text)
    print("Tokenization successful:", tokens)
except Exception as e:
    print("Error during tokenization:", str(e)) 
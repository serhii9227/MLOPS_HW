import nltk
import sys
print("Python version:", sys.version)
print("NLTK version:", nltk.__version__)
print("NLTK path:", nltk.__file__)
print("\nNLTK data path:")
print(nltk.data.path)

try:
    nltk.data.find('tokenizers/punkt')
    print("\nPunkt tokenizer is installed")
except LookupError:
    print("\nPunkt tokenizer is NOT installed")

try:
    nltk.data.find('corpora/stopwords')
    print("Stopwords are installed")
except LookupError:
    print("Stopwords are NOT installed")

# Try to use the tokenizer
text = "This is a test sentence."
try:
    tokens = nltk.word_tokenize(text)
    print("\nTokenization test successful:", tokens)
except Exception as e:
    print("\nTokenization test failed:", str(e)) 
# app/model.py

import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from keras.preprocessing.sequence import pad_sequences
import re
import nltk
from nltk.corpus import stopwords

# Initialize stopwords
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
stop_words.add("rt")  # add rt to remove retweet in dataset (noise)

def precision(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    predicted_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + tf.keras.backend.epsilon())
    return precision

def recall(y_true, y_pred):
    true_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true * y_pred, 0, 1)))
    possible_positives = tf.reduce_sum(tf.round(tf.clip_by_value(y_true, 0, 1)))
    recall = true_positives / (possible_positives + tf.keras.backend.epsilon())
    return recall

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * (p * r) / (p + r + tf.keras.backend.epsilon())

# Загрузка токенізатора
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

model = load_model("my_model.keras", custom_objects={
    "f1": f1,
    "precision": precision,
    "recall": recall
})

# Максимальна довжина (як при тренуванні)
max_length = 26

# remove html entity:
def remove_entity(raw_text):
    entity_regex = r"&[^\s;]+;"
    text = re.sub(entity_regex, "", raw_text)
    return text

# change the user tags
def change_user(raw_text):
    regex = r"@([^ ]+)"
    text = re.sub(regex, "user", raw_text)
    return text

# remove urls
def remove_url(raw_text):
    url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»""'']))"
    text = re.sub(url_regex, '', raw_text)
    return text

# remove unnecessary symbols
def remove_noise_symbols(raw_text):
    text = raw_text.replace('"', '')
    text = text.replace("'", '')
    text = text.replace("!", '')
    text = text.replace("`", '')
    text = text.replace("..", '')
    return text

# remove stopwords
def remove_stopwords(raw_text):
    tokenize = nltk.word_tokenize(raw_text)
    text = [word for word in tokenize if not word.lower() in stop_words]
    text = " ".join(text)
    return text

# Функція очистки тексту
def preprocess(texts):
    clean = []
    # change the @xxx into "user"
    clean = [change_user(text) for text in texts]
    # remove emojis (specifically unicode emojis)
    clean = [remove_entity(text) for text in clean]
    # remove urls
    clean = [remove_url(text) for text in clean]
    # remove trailing stuff
    clean = [remove_noise_symbols(text) for text in clean]
    # remove stopwords
    clean = [remove_stopwords(text) for text in clean]
    return clean

# Головна функція передбачення
def predict_text(text: str) -> dict:
    clean = preprocess([text])
    seq = tokenizer.texts_to_sequences(clean)
    padded = pad_sequences(seq, maxlen=max_length)
    probs = model.predict(padded)[0]
    pred_class = int(probs.argmax())

    label_map = {
        0: 'hate speech',
        1: 'offensive speech',
        2: 'neutral speech'
    }

    return {
        "class": label_map.get(pred_class),
        "confidence": float(probs.max())
    }

# Test the model
if __name__ == "__main__":
    test_text = "I hate  you"
    result = predict_text(test_text)
    print(f"\nTest prediction for text: '{test_text}'")
    print(f"Result: {result}")

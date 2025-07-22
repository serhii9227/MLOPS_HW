import pandas as pd
import numpy as np
import boto3
from io import StringIO
import os

from nltk.corpus import stopwords
import re
from sklearn.model_selection import train_test_split
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import mlflow
from mlflow.pyfunc import load_model
from mlflow.tracking import MlflowClient
from mlflow.exceptions import RestException




bucket_name = "mlops93"
s3_key = "processed_data/combined_data.csv"

s3 = boto3.client("s3")

obj = s3.get_object(Bucket=bucket_name, Key=s3_key)
data = pd.read_csv(StringIO(obj['Body'].read().decode('utf-8')))
        
cols = list(data.columns)
text_col = [col for col in cols if col != 'label'][0]

labels = list(data['label'])
messages = list(data[text_col])

stop_words = set(stopwords.words('english'))


def remove_entity(raw_text):
            entity_regex = r"&[^\s;]+;"
            text = re.sub(entity_regex, "", raw_text)
            return text
        
def change_user(raw_text):
    regex = r"@([^ ]+)"
    text = re.sub(regex, "user", raw_text)
    return text

def remove_url(raw_text):
    url_regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»""'']))"
    text = re.sub(url_regex, '', raw_text)
    return text

def remove_noise_symbols(raw_text):
    text = raw_text.replace('"', '')
    text = text.replace("'", '')
    text = text.replace("!", '')
    text = text.replace("`", '')
    text = text.replace("..", '')
    return text

def remove_stopwords(raw_text):
    try:
        tokenize = nltk.word_tokenize(raw_text)
        text = [word for word in tokenize if not word.lower() in stop_words]
        text = " ".join(text)
        return text
    except:
        return raw_text

def preprocess(datas):
    clean = []
    clean = [change_user(text) for text in datas]
    clean = [remove_entity(text) for text in clean]
    clean = [remove_url(text) for text in clean]
    clean = [remove_noise_symbols(text) for text in clean]
    clean = [remove_stopwords(text) for text in clean]
    return clean


clean_messages = preprocess(messages)
X_train, X_test, y_train, y_test = train_test_split(clean_messages, labels, test_size=0.2, random_state=42)   

TOKENIZER_KEY = "tokenizer.pkl"
response = s3.get_object(Bucket=bucket_name, Key=TOKENIZER_KEY)
tokenizer = pickle.loads(response['Body'].read())


X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(seq) for seq in X_train)
X_train = pad_sequences(X_train, maxlen=max_length)
X_test = pad_sequences(X_test, maxlen=max_length)


y_test = np.array(y_test)
y_train = np.array(y_train)

output_dim = 200


model = Sequential([
    Embedding(vocab_size, output_dim, input_length=max_length),
    LSTM(64, dropout=0.3, recurrent_dropout=0.3),
    Dropout(0.5),
    Dense(128, activation="relu"),
    Dropout(0.5),
    Dense(1, activation="sigmoid"),
])


model.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
)


model_history = model.fit(
    X_train,
    y_train,
    batch_size=64,
    epochs=5,
    validation_data=(X_test, y_test),
    verbose=1
)



y_pred = model.predict(X_test)
y_pred = (y_pred >= 0.5).astype(int)
f1 = f1_score(y_test, y_pred, zero_division=0)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)


MODEL_NAME = "HateSpeechLSTM"
EXPERIMENT_NAME = "Detect hate speech"
BUCKET_NAME = "mlops93"

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()


experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    experiment_id = client.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id

# check existing model
try:
    latest_versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
    first_model = len(latest_versions) == 0
except RestException:
    latest_versions = []
    first_model = True


# Compare with previous model

if not first_model:
    try:
        model_info = client.get_model_version(name=MODEL_NAME, version=latest_versions[0].version)
        artifact_uri = model_info.source  # типу: runs:/<run_id>/model

        local_path = mlflow.artifacts.download_artifacts(artifact_uri)

        model_path = os.path.join(local_path, "model.keras")
        previous_model = load_model(model_path)

        y_pred_prev = previous_model.predict(X_test)
        y_pred_prev = (y_pred_prev >= 0.5).astype(int)
        f1_previous = f1_score(y_test, y_pred_prev, zero_division=0)
    except Exception as e:
        f1_previous = -1
else:
    f1_previous = -1




# Register model
if first_model or f1 > f1_previous:
    with mlflow.start_run(experiment_id=experiment_id) as run:
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        model_path = "model.keras"
        model.save(model_path)

        mlflow.log_artifact(model_path, artifact_path="model")
        
        model_uri = f"runs:/{run.info.run_id}/model"
        
        # create model in register
        try:
            client.create_registered_model(MODEL_NAME)
        except RestException:
            pass
        
        # Register new version
        model_version = client.create_model_version(
            name=MODEL_NAME,
            source=model_uri,
            run_id=run.info.run_id
        )
        
        # Archive previous verion
        for v in client.get_latest_versions(MODEL_NAME, stages=["Production"]):
            if v.version != model_version.version:
                client.transition_model_version_stage(
                    name=MODEL_NAME,
                    version=v.version,
                    stage="Archived"
                )
        
        # Promote new version
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=model_version.version,
            stage="Production"
        )

 # Завантаження артефактів
        local_path = os.path.join("/tmp", "artifacts", f"v{model_version.version}")
        os.makedirs(local_path, exist_ok=True)
        mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=local_path)
        
        # Завантаження до S3
        def upload_dir_to_s3(local_dir, bucket, s3_prefix):
            for root, dirs, files in os.walk(local_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, local_dir)
                    s3_key = os.path.join(s3_prefix, relative_path).replace("\\", "/")
                    s3.upload_file(full_path, bucket, s3_key)
        
        upload_dir_to_s3(local_path, BUCKET_NAME, f"artifacts/v{model_version.version}")
else:
    print("Model is not better")


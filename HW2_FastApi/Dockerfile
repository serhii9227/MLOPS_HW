FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt
RUN python download_all_nltk.py

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
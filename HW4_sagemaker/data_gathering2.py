import os
import boto3
import shutil
import pandas as pd
from io import BytesIO
import io
import subprocess


#Step 1 - gathering data

repo_name = "MLOps_PO"
repo_url = f"https://github.com/serhii9227/{repo_name}.git"
subprocess.run(["git", "clone", repo_url], check=True)
key = 'uploaded_csv.txt'


bucket = 'mlops93'
files = []
for file in os.listdir(repo_name):
    if not file.startswith('.') and file.endswith('.csv') and os.path.isfile(os.path.join(repo_name, file)):
        files.append(file)

s3 = boto3.client('s3')

try:
    response = s3.get_object(Bucket=bucket, Key=key)
    uploaded_files = response['Body'].read().decode('utf-8').splitlines()
except s3.exceptions.NoSuchKey:
    uploaded_files = []

new_files = [file for file in files if file not in uploaded_files]

s3 = boto3.client('s3')
for file in new_files:
    s3.upload_file(os.path.join(repo_name, file), bucket, file)
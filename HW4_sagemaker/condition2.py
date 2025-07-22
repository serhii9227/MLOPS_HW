import os
import boto3
from io import BytesIO

bucket = 'mlops93'
s3 = boto3.client('s3')
response = s3.list_objects_v2(Bucket=bucket)

files = []
if 'Contents' in response:
    for content in response['Contents']:
        if content['Key'].endswith('.csv'):
            files.append(content['Key'])

key = 'uploaded_csv.txt'
try:
    obj = s3.get_object(Bucket=bucket, Key=key)
    uploaded_csv = obj["Body"].readlines()
    uploaded_csv = [line.decode('utf-8').strip() for line in uploaded_csv if line.decode('utf-8').strip().endswith('.csv')]
except s3.exceptions.NoSuchKey:
    uploaded_csv = []

for file in files:
    if file.startswith('processed_data/'):
        continue
    if file not in uploaded_csv:
        uploaded_csv.append(file)

new_content = '\n'.join(uploaded_csv) + '\n'
file_obj = BytesIO(new_content.encode('utf-8'))
s3.upload_fileobj(file_obj, bucket, key)

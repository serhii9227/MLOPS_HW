import os
import boto3
import shutil
import pandas as pd
from io import BytesIO
import io


#Step 3 - process data

bucket = 'mlops93'
folder = 'processed_data/'
s3 = boto3.client('s3')

response = s3.list_objects_v2(Bucket=bucket, Prefix=folder)
processed_csv = []
for content in response['Contents']:
    file = content['Key'].replace('processed_data/', '')
    if file.endswith('.csv') and file != 'combined_data.csv':
        processed_csv.append(file)

key_txt = 'uploaded_csv.txt'
obj = s3.get_object(Bucket=bucket, Key=key_txt)
uploaded_csv = obj["Body"].readlines()
uploaded_csv = [line.decode('utf-8').strip() for line in uploaded_csv if line.decode('utf-8').strip().endswith('.csv')]

non_processed_csv = []
for csv in uploaded_csv:
    if not csv in processed_csv:
        non_processed_csv.append(csv)
key = f'{folder}combined_data.csv'
obj = s3.get_object(Bucket=bucket, Key=key)

df_combined_data = pd.read_csv(io.StringIO(obj['Body'].read().decode('utf-8')))


for csv in non_processed_csv:
    key = csv
    obj = s3.get_object(Bucket=bucket, Key=key)
    df = pd.read_csv(io.StringIO(obj['Body'].read().decode('utf-8')))

    df.columns = [col[0].lower() + col[1:] if col else col for col in df.columns]
    df_new = pd.DataFrame()
    
    name_col_1 = ['text', 'content', 'tweet']
    for col in name_col_1:
        if col in df.columns:
            df_new['tweet'] = df[col]
    
    if 'label' in df.columns:
        df_new['label'] = df['label']
    
    label_values = (df_new['label'].unique())
    
    def safe_to_int(val):
        try:
            return int(val)
        except (ValueError, TypeError):
            return val 
    
    df_new['label'] = df_new['label'].apply(safe_to_int)

    if 1 and 0 in label_values:
        pass
    
    else:
        if 'hate' in label_values:
            df_new['label'] = df_new['label'].replace('hate', 1)
    
        if 'nothate' in label_values:
            df_new['label'] = df_new['label'].replace('nothate', 0)

    try:
        df_new = df_new[df_new['label'].isin([0, 1])].copy()
    except (ValueError, TypeError):
        print("Error: label column contains non-numeric values")

    if len(df_new[df_new["label"] == 0]) > len(df_new[df_new["label"] == 1]):
        df_major = df_new[df_new["label"] == 0]
        df_minor = df_new[df_new["label"] == 1]
    
    elif len(df_new[df_new["label"] == 0]) < len(df_new[df_new["label"] == 1]):
        df_major = df_new[df_new["label"] == 1] 
        df_minor = df_new[df_new["label"] == 0]
    
    else:
        pass
    
    
    df_major_downsampled = df_major.sample(n=len(df_minor), random_state=42)
    
    df_balanced = pd.concat([df_major_downsampled, df_minor])
    
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    df_combined_data = pd.concat([df_combined_data, df_balanced], ignore_index=True)
    
    csv_buffer = io.StringIO()
    df_balanced.to_csv(csv_buffer, index=False)
    s3.put_object(Bucket=bucket, Key=f'{folder}{csv}', Body=csv_buffer.getvalue())


s3 = boto3.client('s3')
s3.delete_object(Bucket=bucket, Key='processed_data/combined_data.csv')

csv_buffer = io.StringIO()
df_combined_data.to_csv(csv_buffer, index=False)
s3.put_object(Bucket=bucket, Key='processed_data/combined_data.csv', Body=csv_buffer.getvalue())
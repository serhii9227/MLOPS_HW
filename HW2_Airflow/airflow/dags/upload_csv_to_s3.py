import os
import boto3
from datetime import datetime
from airflow import DAG
from airflow.providers.standard.operators.python import PythonOperator



def find_new_csv(**kwargs):

    files = []
    for file in os.listdir('data'):
        if not file.startswith('.') and file.endswith('.csv') and os.path.isfile(os.path.join('data', file)):
            files.append(file)

    with open('/opt/airflow/uploaded_csv.txt', 'r') as file:
        uploaded_files = file.readlines()

    new_files = [file for file in files if file not in uploaded_files]
    kwargs['ti'].xcom_push(key='new_files', value=new_files)

def upload_csv_to_s3(**kwargs):
    ti = kwargs['ti']
    new_files = ti.xcom_pull(key='new_files', task_ids='find_new_csv')

    s3 = boto3.client('s3')
    for file in new_files:
        s3.upload_file(os.path.join('data', file), 'mlops92', file)
        with open('/opt/airflow/uploaded_csv.txt', 'a') as f:
            f.write(file + '\n')

with DAG(
    dag_id='upload_csv_to_s3',
    start_date=datetime(2025, 7, 1),
    schedule='@daily',
    catchup=False
) as dag:

    find_new_csv_task = PythonOperator(
        task_id='find_new_csv',
        python_callable=find_new_csv
    )

    upload_csv_to_s3_task = PythonOperator(
        task_id='upload_csv_to_s3',
        python_callable=upload_csv_to_s3
    )

    find_new_csv_task >> upload_csv_to_s3_task

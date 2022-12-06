import pandas as pd
from sqlalchemy import create_engine
import os
import s3_management
import boto3

host=       os.environ.get('MODEL_DB_HOST')
port=       os.environ.get('MODEL_DB_PORT')
user=       os.environ.get('MODEL_DB_USER')
password =  os.environ.get('MODEL_DB_PASS')
bucket_name = os.environ.get('BUCKET_NAME')

engine = create_engine(f'postgresql://{user}:{password}@{host}:5432/postgres')
print('engine created')


def generate_csv(date):
    carpeta='data/to_train/'
    nombre_archivo = 'credit_card_clients' + date + '.csv.gz'
    #download table credit_card_clients from database to csv
    df = pd.read_sql_table('credit_card_clients', engine)
    print('dataframe created')
    df.to_csv(carpeta+'/'+nombre_archivo, compression='gzip', index=False)
    print('dataframe saved')
    return nombre_archivo

def load_csv_to_s3(date):
    carpeta='data/to_train/'
    folder_name = 'to_train'
    s3_client = boto3.client("s3")
    s3_resource = boto3.resource("s3")
    
    if s3_resource.Bucket(bucket_name).creation_date is None:
        s3_client.create_bucket(Bucket=bucket_name)
    
    #validate if folder exists
    if(not s3_management.validate_if_folder_exist(bucket_name, folder_name)):
        if(not s3_management.validate_if_bucket_exist(bucket_name)):
            s3_management.create_bucket(bucket_name)
        s3_management.create_folder(bucket_name, folder_name)
        
    nombre_archivo = generate_csv(date)    
    try:
        s3_management.upload_csv_file(carpeta+'/'+nombre_archivo, bucket_name, folder_name+'/'+nombre_archivo)
        print(f'Archivo {nombre_archivo} cargado, en carpeta {folder_name}')
    except Exception as e:
        print(f'Error al cargar archivo {nombre_archivo}')
        print(e)
        
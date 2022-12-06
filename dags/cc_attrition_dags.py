from datetime import datetime
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy_operator import DummyOperator

from db_credit_card_client import load_csv_to_s3
from train_model import train_model
import predict
import load_client_prediction

def start():
    print('Starting the DAG')
    
    
def download_credit_card_data(**context):
    date = f"{context['logical_date']:%Y%m%d}"
    print('Downloading credit card clients')
    load_csv_to_s3(date)
    
def train_credict_card_model(**context):
    date = f"{context['logical_date']:%Y%m%d}"
    print('Training model')
    train_model(date)
    
def predict_credict_card_attrition(**context):
    print('Predicting')    
    date = f"{context['logical_date']:%Y%m%d}"
    data=predict.read_data(date)
    data_to_be_predicted_union = predict.prepare_data(data)
    new_data_predicted=predict.predict_attrition_flag(data_to_be_predicted_union,data)
    predict.save_to_csv(new_data_predicted,date)
    predict.delete_local_file(date)

def load_attried_clients_probability(**context):
    print('Loading attried clients probability')    
    date = f"{context['logical_date']:%Y%m%d}"
    load_client_prediction.load_credit_card_clients_data(date)
    

default_args = {
    'owner': 'patricio',
    'retries': 0,
    'start_date': datetime(2022, 12, 2),
}
with DAG(
    dag_id='credit_card_clients',
    default_args=default_args,
    schedule_interval='5 4 * * TUE-SAT',
    tags=['credit_card_clients'],
    max_active_runs=1
) as dag:
    start_dummy = DummyOperator(task_id='start')
    download_credit_card_data = PythonOperator(
        task_id='download_credit_card_data',
        python_callable=download_credit_card_data,
        provide_context=True,
    )
    train_credict_card_model = PythonOperator(
        task_id='train_credict_card_model',
        python_callable=train_credict_card_model,
        provide_context=True,
    )
    predict_credict_card_attrition = PythonOperator(
        task_id='predict_credict_card_attrition',
        python_callable=predict_credict_card_attrition,
        provide_context=True,
    )
    load_attried_clients_probability = PythonOperator(
        task_id='load_attried_clients_probability',
        python_callable=load_attried_clients_probability,
        provide_context=True,      
    )
        
    end_dummy = DummyOperator(task_id='end')
    start_dummy >> download_credit_card_data >> train_credict_card_model >> predict_credict_card_attrition >> load_attried_clients_probability >> end_dummy
    start_dummy >> end_dummy
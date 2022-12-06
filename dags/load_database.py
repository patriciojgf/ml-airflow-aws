import pandas as pd
from sqlalchemy import create_engine

host='creditcardclientattrition.cgxic12usgz2.us-east-1.rds.amazonaws.com'
port='5432'
user='postgres'
password='4dadPxHWdr7bK#3YczPw'
engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/postgres')
print('engine created')


def load_credit_card_clients_data(date):
    nombre_archivo = f'credit_card_clients{date}.csv.gz'
    archivo = 'data/predicted/'+nombre_archivo
    df = pd.read_csv(archivo, compression='gzip', sep=';')
    #rename columns
    df['Naive_Bayes_Classifier_Attrition_Flag_1'] =\
        df['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1']
    df['Naive_Bayes_Classifier_Attrition_Flag_2'] =\
        df['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2']
    df.drop(['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',\
                'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'],\
                inplace=True, axis=1
    )   
    
    #convert to float64    
    df['Credit_Limit'] = df['Credit_Limit'].astype('float64')
    df['Avg_Open_To_Buy'] = df['Avg_Open_To_Buy'].astype('float64')
    #convert to float64
    df['Naive_Bayes_Classifier_Attrition_Flag_1'] = df['Naive_Bayes_Classifier_Attrition_Flag_1'].str.replace(',','.').astype('float64')

    #delete from database where date is equal to date with sqlalchemy
    try:
        engine.execute(f"DELETE FROM credit_card_clients_attried WHERE date = '{date}'")
        print(f'Archivo {nombre_archivo} eliminado de la base de datos')
    except Exception as e:
        print(f'Error al eliminar archivo {nombre_archivo} de la base de datos')
        print(e)
    
    
    try:
        df.to_sql('credit_card_clients_attried', engine, if_exists='append', index=False)
        print(f'Archivo {nombre_archivo} cargado')
    except Exception as e:
        print(f'Error al cargar archivo {nombre_archivo}')
        print(e)
    
    
import pandas as pd
from sqlalchemy import create_engine
import os

host=       os.environ.get('MODEL_DB_HOST')
port=       os.environ.get('MODEL_DB_PORT')
user=       os.environ.get('MODEL_DB_USER')
password=   os.environ.get('MODEL_DB_PASS')

engine = create_engine(f'postgresql://{user}:{password}@{host}:5432/postgres')
print('engine created')


def load_credit_card_clients_data(date):
    nombre_archivo = f'credit_card_clients{date}.csv.gz'
    archivo = 'data/predicted/'+nombre_archivo
    df = pd.read_csv(archivo, compression='gzip', sep=';')
    
    
    df2 = df[(df['Attrition_Flag'] == 'Attrited Customer') & (df['Attrition_Flag_Probability'] > 50)]
    df2=df2[['CLIENTNUM','Attrition_Flag_Probability']]
    df2['date']=date
    
    try:
        resultado=engine.execute(f"DELETE FROM attried_clients_probability WHERE date = '{date}'")
        print(f'deleted rows: ',resultado.rowcount)
    except Exception as e:
        print(f'Error al eliminar archivo {nombre_archivo} de la base de datos')
        print(e)    
    
    try:
        df2.to_sql('attried_clients_probability' , engine, if_exists='append', index=False)
        print(f'Archivo {nombre_archivo} cargado')
    except Exception as e:
        print(f'Error al cargar archivo {nombre_archivo}')
        print(e)
        

    #delete local file
    try:
        os.remove(archivo)
        print(f'Archivo {nombre_archivo} eliminado')
    except Exception as e:
        print(f'Error al eliminar archivo {nombre_archivo}')
        print(e)
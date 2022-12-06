import pandas as pd
from sqlalchemy import create_engine

host='creditcardclientattrition.cgxic12usgz2.us-east-1.rds.amazonaws.com'
port='5432'
user='postgres'
password='4dadPxHWdr7bK#3YczPw'

engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/postgres')
# engine = create_engine('postgresql://postgres:postgres@creditcardclientattrition.cgxic12usgz2.us-east-1.rds.amazonaws.com
# :5432/postgres')
print('engine created')

#read model_database/raw_data/BankChurners.csv.zip to a pandas dataframe
df = pd.read_csv('model_database/raw_data/BankChurners.csv.gz', compression='gzip',delimiter=',', quotechar='"')
print('dataframe created')
#rename columns
df['Naive_Bayes_Classifier_Attrition_Flag_1'] =\
    df['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1']
df['Naive_Bayes_Classifier_Attrition_Flag_2'] =\
    df['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2']
df.drop(['Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',\
            'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'],\
            inplace=True, axis=1
)   
print('columns renamed')
#create table
df.dtypes
df.to_sql('credit_card_clients', engine, if_exists='replace', index=False)
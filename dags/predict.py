import pandas  as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import os
import s3_management
import pickle
bucket_name = os.environ.get('BUCKET_NAME')

def read_data(date):
    carpeta_local = 'data/to_predict/'
    carpeta_s3 = 'to_predict/'
    nombre_archivo = 'data_to_be_predicted' + date + '.csv'
    s3_management.download_file(bucket_name, carpeta_s3+nombre_archivo, carpeta_local+nombre_archivo)
    data = pd.read_csv(carpeta_local+nombre_archivo, delimiter=';', quotechar='"')
    return data

def delete_local_file(date):
    nombre_archivo = 'data/to_predict/data_to_be_predicted' + date + '.csv'
    os.remove(nombre_archivo)
    print('File deleted from local')

def prepare_data(data):
    data_to_be_predicted =data.drop(['CLIENTNUM','Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1',\
                'Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2'],\
                inplace=False, axis=1)
    data_to_be_predicted["Gender"]=data_to_be_predicted["Gender"].map({'M': 1, 'F': 0})
    categoricos_data_to_be_predicted = data_to_be_predicted.loc[:,data_to_be_predicted.dtypes==np.object]
    
    categoricos_data_to_be_predicted2 = categoricos_data_to_be_predicted.reset_index()
    categoricos_data_to_be_predicted2.drop(['index'], axis = 'columns', inplace=True) 
    
    numericos_data_to_be_predicted = data_to_be_predicted.loc[:,data_to_be_predicted.dtypes!=np.object]
    #Robust standardization for training set 
    scaler = RobustScaler() 
    data_scaled = scaler.fit_transform(numericos_data_to_be_predicted)
    new = pd.DataFrame(data_scaled)
    datos_data_to_be_predicted_num_standar = new.rename({0: 'Customer_Age', 1: 'Gender',2:'Dependent_count', 3: 'Months_on_book', 4: 'Total_Relationship_Count', 5: 'Months_Inactive_12_mon', 6: 'Contacts_Count_12_mon', 7: 'Credit_Limit', 8: 'Total_Revolving_Bal', 9: 'Avg_Open_To_Buy', 10: 'Total_Amt_Chng_Q4_Q1', 11: 'Total_Trans_Amt', 12: 'Total_Trans_Ct', 13: 'Total_Ct_Chng_Q4_Q1', 14: 'Avg_Utilization_Ratio'}, axis=1)

    union = pd.concat([datos_data_to_be_predicted_num_standar,categoricos_data_to_be_predicted2], axis = "columns")

    #Convert categorical data to numeric
    data_to_be_predicted_union =pd.get_dummies(data=union, drop_first=True)
    
    return data_to_be_predicted_union

def predict_attrition_flag(data_to_be_predicted_union,data):
    #Load trained model from dags/model/trained_model.pkl and predict
    with open('dags/model/trained_model.pkl', 'rb') as f:
        best_model = pickle.load(f)
        pred_xgb_best = best_model.predict(data_to_be_predicted_union)    
        #calculate probability of attrition as porcentage
        proba_xgb_best = best_model.predict_proba(data_to_be_predicted_union)[:,1]*100
        
    
     
    #add prediction to data_to_be_predicted as new column "Attrition_Flag"
    new_data_to_be_predicted = data.copy()
    new_data_to_be_predicted['Attrition_Flag'] = pred_xgb_best
    
    new_data_to_be_predicted['Attrition_Flag_Probability'] = proba_xgb_best
    new_data_to_be_predicted["Attrition_Flag"]=new_data_to_be_predicted["Attrition_Flag"].map({0:'Existing Customer',1: 'Attrited Customer'})
    
    return new_data_to_be_predicted

def save_to_csv(new_data_to_be_predicted,date):
    nombre_archivo = 'data/predicted/credit_card_clients' + date + '.csv.gz'
    new_data_to_be_predicted.to_csv(nombre_archivo, index=False, compression='gzip', sep=';')
    print('File saved to S3')
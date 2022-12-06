import warnings
import statistics as stat
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score,f1_score,classification_report,precision_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
import xgboost 
import statistics as stat
import pickle
import s3_management
import os

bucket_name = os.environ.get('BUCKET_NAME')


def train_model(date):
    #agregar que reciba el archivo por parametro.
    nombre_archivo = 'credit_card_clients' + date + '.csv.gz'
        
    #read data/credit_card_clients20221202.csv.gz to a pandas dataframe
    df_file = pd.read_csv('data/to_train/'+nombre_archivo, compression='gzip',delimiter=',', quotechar='"')
    #df_file.drop(['CLIENTNUM', 'Unnamed: 21'], axis = 'columns', inplace=True)
    df_file.drop(['CLIENTNUM','Naive_Bayes_Classifier_Attrition_Flag_1','Naive_Bayes_Classifier_Attrition_Flag_2'], axis = 'columns', inplace=True)
    #Columns transformation
    df_file["Attrition_Flag"]=df_file["Attrition_Flag"].map({'Existing Customer': 0, 'Attrited Customer': 1})
    df_file["Gender"]=df_file["Gender"].map({'M': 1, 'F': 0})

    #Split in training data and test data
    objetivo = df_file["Attrition_Flag"]
    data = df_file.drop(columns=["Attrition_Flag"])
    datos_entrena, datos_prueba, clase_entrena, clase_prueba = train_test_split(data,objetivo,test_size=0.30,random_state=41)
    df = datos_entrena
    df2 = datos_prueba


    #Separate numerical and categorical data
    cat_entreamiento = df.loc[:,df.dtypes==np.object]
    cat_entreamiento2 = cat_entreamiento.reset_index()
    cat_entreamiento2.drop(['index'], axis = 'columns', inplace=True) 

    cat_test = df2.loc[:,df2.dtypes==np.object]
    cat_test2 = cat_test.reset_index()
    cat_test2.drop(['index'], axis = 'columns', inplace=True) 


    num_entreno = df.loc[:,df.dtypes!=np.object]
    num_entreno_2 = num_entreno

    num_test = df2.loc[:,df2.dtypes!=np.object]
    num_test_2 = num_test


    #Robust standardization for numerical data
    #Robust standardization for training set 
    scaler = RobustScaler() 
    scaled_num_entreno_2 = scaler.fit_transform(num_entreno_2)
    new = pd.DataFrame(scaled_num_entreno_2)
    df_scaled_num_entreno_2 = new.rename({0: 'Customer_Age', 1: 'Gender',2:'Dependent_count', 3: 'Months_on_book', 4: 'Total_Relationship_Count', 5: 'Months_Inactive_12_mon', 6: 'Contacts_Count_12_mon', 7: 'Credit_Limit', 8: 'Total_Revolving_Bal', 9: 'Avg_Open_To_Buy', 10: 'Total_Amt_Chng_Q4_Q1', 11: 'Total_Trans_Amt', 12: 'Total_Trans_Ct', 13: 'Total_Ct_Chng_Q4_Q1', 14: 'Avg_Utilization_Ratio'}, axis=1)

    #Robust standardization for test set 
    scaler2 = RobustScaler()
    scaled_num_test_2 = scaler2.fit_transform(num_test_2)
    new2 = pd.DataFrame(scaled_num_test_2)
    df_scaled_num_test_2 = new2.rename({0: 'Customer_Age', 1: 'Gender',2:'Dependent_count', 3: 'Months_on_book', 4: 'Total_Relationship_Count', 5: 'Months_Inactive_12_mon', 6: 'Contacts_Count_12_mon', 7: 'Credit_Limit', 8: 'Total_Revolving_Bal', 9: 'Avg_Open_To_Buy', 10: 'Total_Amt_Chng_Q4_Q1', 11: 'Total_Trans_Amt', 12: 'Total_Trans_Ct', 13: 'Total_Ct_Chng_Q4_Q1', 14: 'Avg_Utilization_Ratio'}, axis=1)

    #Union of categorical data with numerical data
    df_union_entrenamiento_cat_num = pd.concat([df_scaled_num_entreno_2,cat_entreamiento2], axis = "columns")
    df_union_test_cat_num = pd.concat([df_scaled_num_test_2,cat_test2], axis = "columns")
    #Convert categorical data to numeric
    datos_entrena_union =pd.get_dummies(data=df_union_entrenamiento_cat_num, drop_first=True)
    datos_test_union =pd.get_dummies(data=df_union_test_cat_num, drop_first=True)

    #Data balance
    smote = SMOTE()
    datos_entrena_balance, clase_entrena_balance = smote.fit_resample(datos_entrena_union, clase_entrena) 
    datos_test_balance, clase_test_balance= smote.fit_resample(datos_test_union,clase_prueba)

    #Clean outliers for training data 
    test = datos_entrena_balance["Contacts_Count_12_mon"]
    Q3 = test.quantile(0.75) #max
    Q1 = test.quantile(0.25) #min
    IQR = Q3 - Q1
    min_value = Q1 - (1.5*IQR)
    max_value = Q3 + (1.5*IQR)
    test = np.where(test>max_value,1,test)
    test = np.where(test<min_value,1,test)
    datos_entrena_balance["Contacts_Count_12_mon"] = test

    test = datos_test_balance["Contacts_Count_12_mon"]
    Q3 = test.quantile(0.75) #max
    Q1 = test.quantile(0.25) #min
    IQR = Q3 - Q1
    min_value = Q1 - (1.5*IQR)
    max_value = Q3 + (1.5*IQR)
    test = np.where(test>max_value,1,test)
    test = np.where(test<min_value,1,test)
    datos_test_balance["Contacts_Count_12_mon"] = test

    test = datos_entrena_balance["Credit_Limit"]
    moda=stat.mode(test)
    Q3 = test.quantile(0.75) #max
    Q1 = test.quantile(0.25) #min
    IQR = Q3 - Q1
    min_value = Q1 - (1.5*IQR)
    max_value = Q3 + (1.5*IQR)
    test = np.where(test>max_value,moda,test)
    datos_entrena_balance["Credit_Limit"] = test

    test = datos_test_balance["Credit_Limit"]
    moda=stat.mode(test)
    Q3 = test.quantile(0.75) #max
    Q1 = test.quantile(0.25) #min
    IQR = Q3 - Q1
    min_value = Q1 - (1.5*IQR)
    max_value = Q3 + (1.5*IQR)
    test = np.where(test>max_value,moda,test)
    datos_test_balance["Credit_Limit"] = test


    test = datos_entrena_balance["Total_Amt_Chng_Q4_Q1"]
    moda=stat.mode(test)
    Q3 = test.quantile(0.75) #max
    Q1 = test.quantile(0.25) #min
    IQR = Q3 - Q1
    min_value = Q1 - (1.5*IQR)
    max_value = Q3 + (1.5*IQR)
    test = np.where(test>max_value,moda,test)
    test = np.where(test<min_value,moda,test)
    datos_entrena_balance["Total_Amt_Chng_Q4_Q1"] = test

    test = datos_test_balance["Total_Amt_Chng_Q4_Q1"]
    Q3 = test.quantile(0.75) #max
    Q1 = test.quantile(0.25) #min
    IQR = Q3 - Q1
    min_value = Q1 - (1.5*IQR)
    max_value = Q3 + (1.5*IQR)
    test = np.where(test>max_value,moda,test)
    test = np.where(test<min_value,moda,test)
    datos_test_balance["Total_Amt_Chng_Q4_Q1"] = test

    test = datos_entrena_balance["Total_Trans_Amt"]
    Q3 = test.quantile(0.75) #max
    Q1 = test.quantile(0.25) #min
    IQR = Q3 - Q1
    min_value = Q1 - (1.5*IQR)
    max_value = Q3 + (1.5*IQR)
    test = np.where(test>max_value,moda,test)
    test = np.where(test<min_value,moda,test)
    datos_entrena_balance["Total_Trans_Amt"] =test

    test = datos_test_balance["Total_Trans_Amt"]
    Q3 = test.quantile(0.75) #max
    Q1 = test.quantile(0.25) #min
    IQR = Q3 - Q1
    min_value = Q1 - (1.5*IQR)
    max_value = Q3 + (1.5*IQR)
    test = np.where(test>max_value,moda,test)
    test = np.where(test<min_value,moda,test)
    datos_test_balance["Total_Trans_Amt"] =test

    test = datos_entrena_balance["Total_Trans_Ct"]
    Q3 = test.quantile(0.75) #max
    Q1 = test.quantile(0.25) #min
    IQR = Q3 - Q1
    min_value = Q1 - (1.5*IQR)
    max_value = Q3 + (1.5*IQR)
    test = np.where(test>max_value,moda,test)
    datos_entrena_balance["Total_Trans_Ct"] = test


    test = datos_test_balance["Total_Trans_Ct"]
    Q3 = test.quantile(0.75) #max
    Q1 = test.quantile(0.25) #min
    IQR = Q3 - Q1
    min_value = Q1 - (1.5*IQR)
    max_value = Q3 + (1.5*IQR)
    test = np.where(test>max_value,moda,test)
    datos_test_balance["Total_Trans_Ct"] = test

    test = datos_entrena_balance["Total_Ct_Chng_Q4_Q1"]
    Q3 = test.quantile(0.75) #max
    Q1 = test.quantile(0.25) #min
    IQR = Q3 - Q1
    min_value = Q1 - (1.5*IQR)
    max_value = Q3 + (1.5*IQR)
    test = np.where(test>max_value,moda,test)
    test = np.where(test<min_value,moda,test)
    datos_entrena_balance["Total_Ct_Chng_Q4_Q1"] = test

    test = datos_test_balance["Total_Ct_Chng_Q4_Q1"]
    Q3 = test.quantile(0.75) #max
    Q1 = test.quantile(0.25) #min
    IQR = Q3 - Q1
    min_value = Q1 - (1.5*IQR)
    max_value = Q3 + (1.5*IQR)
    test = np.where(test>max_value,moda,test)
    test = np.where(test<min_value,moda,test)
    datos_test_balance["Total_Ct_Chng_Q4_Q1"] = test

    #Train model Xgboost
    xgb = xgboost.XGBClassifier()
    xgb.fit(datos_entrena_balance, clase_entrena_balance)
    #Tunning Xgboost
    # parametros = {
    #                 'max_depth' : [2,4,6,8,10,12],
    #                 'subsample' : [0.1, 0.3, 0.5,0.8,1],
    #                 'colsample_bylevel' : [0.1,0.4,0.6,1],
    #                 'colsample_bytree'  : [0.1,0.4,0.6,1],
    #                 'min_child_weight' : [0.1,0.5,1],
    #                 'n_estimators' : [10, 100,200, 500, 800],
    #                 'learning_rate' : [0.01,0.3,0.6]
    # }
    parametros2 = {
                    'max_depth' : [2,3,5,10,15],
                    'subsample' : [0.1, 0.3, 0.5,0.8,1],
                    'min_child_weight' : [1,2,3,4],
                    'n_estimators' : [100,500, 900, 1100,1500],
                    'learning_rate' : [0.05,0.1,0.15,0.20],
                    #'booster' : ['gbtree','gblinear']
    }

    #Apply RandomizedSearchCV
    warnings.filterwarnings("ignore")
    random_cv = RandomizedSearchCV(xgb,parametros2,cv=5,scoring="f1_macro", n_iter=5,random_state=42)
    fit_obj = random_cv.fit(datos_entrena_balance, clase_entrena_balance)
    best_model = fit_obj.best_estimator_

    #Training best model
    best_model.fit(datos_entrena_balance, clase_entrena_balance)
    pred_xgb_best = best_model.predict(datos_test_balance)

    #Calculate Accuracy Score
    score_xgb = xgb.score(datos_test_balance, clase_test_balance)
    print('score_xgb: ',score_xgb)

    #Calculate Accuracy Score
    accuracy_score_xgb = accuracy_score(clase_test_balance,pred_xgb_best)
    print('accuracy_score_xgb: ',accuracy_score_xgb)

    #Calculate F1 Score
    f1_xgb = f1_score(clase_test_balance,pred_xgb_best, average= "macro")
    print('f1_xgb: ',f1_xgb)

    #Calculate Classification Report
    x_xgb= classification_report(clase_test_balance,pred_xgb_best)
    print(x_xgb)

    #Calculate Precision Score
    print('Precision Score: ',precision_score(clase_test_balance,pred_xgb_best, average= "macro"))

    #Cross Validation
    warnings.filterwarnings("ignore")
    scores_xgb = cross_val_score(xgb,datos_entrena_balance, clase_entrena_balance, cv=5, scoring="f1_macro")

    #Score Mean
    print('scores_xgb mean: ',scores_xgb.mean())

    #Roc Curve
    r_probs = [0 for _ in range(len(clase_test_balance))]
    x_xgb = xgb.predict_proba(datos_test_balance)
    x_xgb = x_xgb[:, 1]
    r_auc = roc_auc_score(clase_test_balance, r_probs)
    xgb_auc = roc_auc_score(clase_test_balance, x_xgb)
    print('Random (chance) Prediction: AUROC = %.3f' % (r_auc))
    print('Xgboost: AUROC = %.3f' % (xgb_auc))


    #Save model to dags/model/trained_model.pkl
    with open('dags/model/trained_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)

    # #Save model to S3 Bucket
    try:
        s3_management.upload_csv_file('dags/model/trained_model.pkl', bucket_name, 'trained_model/trained_model.pkl')
        print(f'Archivo trained_model.pkl cargado, en carpeta model')
    except Exception as e:
        print(f'Error al cargar archivo trained_model.pkl')
        print(e)
            
    #delete file nombre_archivo
    try:
        os.remove('data/to_train/'+nombre_archivo)
        print(f'Archivo data/to_train/'+nombre_archivo+' eliminado')
    except Exception as e:
        print(f'Error al eliminar archivo '+nombre_archivo)
        print(e)
    
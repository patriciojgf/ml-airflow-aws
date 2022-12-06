<p align="center"> 
    <h1 align="center">Trabajo practico ML Airflow:</h1>
    
<p align="center">
        <h2 align="center">Credit Card Client Attried prediction</h2>
</p>
    
  <p align="center">
    <br>Welcome to my project on customer churn prediction for a bank. 
    <br>This project uses historical data on customers and their credit card accounts, stored in a PostgreSQL database on AWS RDS, to train a machine learning model that predicts the likelihood of a customer switching to a different bank. The model is trained using this dataset, and can be applied to new data in the form of a CSV file stored in an S3 bucket. 
    <br>The goal of this project is to provide the bank with insights on which customers are at risk of switching, so that the bank can take action to retain them.
    <br>
  </p>
</p>


# Table of contents
- [Project structure](#Project-structure)
- [Daily batch process summary](#Daily-batch-process-summary)
- [Results](#Results)
- [Dags](#Dags)
- [Proposed AWS architecture](#Proposed-AWS-architecture)
- [S3 Bucket and Folders](#S3-Bucket-and-Folders)
- [VPC Network Structure](#VPC-Network-Structure)

# Project structure

```text
.
├── README.me
├── dags
│   ├── __init__.py
│   ├── cc_attrition_dags.py
│   ├── db_credit_card_client.py
│   ├── load_client_prediction.py
│   ├── model
│   │   └── trained_model.pkl
│   ├── predict.py
│   ├── s3_management.py
│   └── train_model.py
├── data
│   ├── predicted
│   ├── to_predict
│   │   └── trained_model.pkl
│   └── to_train
├── docker-compose.yaml
├── model_database
│   ├── docker-compose.yaml
│   ├── load_tables.py
│   └── raw_data
│       ├── BankChurners.csv
│       └── BankChurners.csv.gz
├── pg_data
└── plugins

``` 
# Daily batch process summary
The Apache Airflow DAG first downloads the credit_card_clients table from the PostgreSQL database, and uses the data to generate a CSV file. In the next step, the CSV file is used to train a machine learning model for predicting the Attrition_Flag and its probability for bank customers.

The trained model is then applied to a dataset of customers loaded from an S3 bucket, to calculate the Attrition_Flag and its probability for each customer. Finally, this prediction is loaded into the attried_clients_probabily table, where the probable customers who may switch banks are stored for each date.

The daily batch process would be as follows:

- The bank's overnight batch process starts, and information on cards and clients is downloaded from the physical model.
- This historical information is used to train a machine learning model that will predict the Attrition_Flag for each customer.
- The Attrition_Flag is predicted for an active client CSV file that is loaded daily into an S3 bucket.
- The probable customers with a probability of switching banks are loaded into a table in the physical model.

# Results
- ![Visual](/infra/img/results.png)

# Dags

![Visual](/infra/img/dags.png)

- download_credit_card_data: 
  - In this first step, the historical information from the database is downloaded, and the file that will be used to train the model is generated.
- train_credict_card_model: 
  - The model is trained and saved in a PKL file.

- predict_credict_card_attrition: 
  - The file of clients that we want to predict with the trained model is downloaded, the Attrition_Flag is calculated, and the results are saved in a file.
 
- load_attried_clients_probabily: 
  - The results are loaded into the 'attried_clients_probabily' table.

# Proposed AWS architecture
- ![Visual](/infra/img/infra.png)

# S3 Bucket and Folders
-  tp-itba-2022-ml-airflow
   -  initial_database_load   -  model
   -  predicted
   -  to_predict
   -  to_train

# VPC Network Structure
- VPC: 
  - IPv4 CIDR: 172.30.0.0/16

- Public Subnet: 
    - ml-cc-airflow-01: 172.30.7.0/24

- Private Subnets:
  - RDS Subnet
    - rds_private_01: 172.30.8.0/24
    - rds_private_02: 172.30.9.0/24

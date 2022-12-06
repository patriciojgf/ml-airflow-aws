import logging
import os
import boto3
from botocore.exceptions import ClientError

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)


# Set the environment variables
access_key = os.environ.get('AWS_ACCESS_KEY_ID')
secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
session_token = os.environ.get('AWS_SESSION_TOKEN')
region = os.environ.get('AWS_REGION')

# Create an S3 client
def create_client():
    """Create an S3 client"""
    try:
        s3_client = boto3.client(
            service_name='s3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_session_token=session_token            
        )
        logging.info('S3 client created')
    except ClientError as e:
        logging.error(e)
        return None
    return s3_client

# Create an S3 resource
def create_resource():
    """Create an S3 resource"""
    try:
        s3_resource = boto3.resource(
            service_name='s3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            aws_session_token=session_token       
        )
        logging.info('S3 resource created')
    except ClientError as e:
        logging.error(e)
        return None
    return s3_resource

# Create a bucket
def create_bucket(bucket_name):
    """Create an S3 bucket in a specified region

    If a region is not specified, the bucket is created in the S3 default
    region (us-east-1).

    :param bucket_name: Bucket to create
    :return: True if bucket created, else False
    """

    # Create bucket
    try:
        s3_client = create_client()
        location = {'LocationConstraint': region}
        s3_client.create_bucket(
            Bucket=bucket_name,
            CreateBucketConfiguration=location
        )
        logging.info(f'Bucket {bucket_name} created')
    except ClientError as e:
        logging.error(e)
        return False
    return True

#create folder inside bucket
def create_folder(bucket_name, folder_name):
    """Create a folder inside a bucket

    :param bucket_name: Bucket to create folder
    :param folder_name: Folder to create
    :return: True if folder was created, else False
    """
    s3_resource = create_resource()
    try:
        s3_resource.Object(bucket_name, folder_name + '/').put()
    except ClientError as e:
        logging.error(e)
        return False
    return True

def upload_csv_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = file_name

    # Upload the file
    s3_client = create_client()
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

def download_file(bucket, object_name, file_name):
    """Download a file from an S3 bucket

    :param bucket: Bucket to download from
    :param object_name: S3 object name. If not specified then file_name is used
    :param file_name: File to download
    :return: True if file was downloaded, else False
    """

    # Download the file
    s3_client = create_client()
    try:
        response = s3_client.download_file(bucket, object_name, file_name)
        print(f'File {file_name} downloaded')
    except ClientError as e:
        logging.error(e)
        return False
    return True


def validate_if_bucket_exist(bucket_name):
    """Validate if a bucket exist

    :param bucket_name: Bucket to validate
    :return: True if bucket exist, else False
    """
    s3_client = create_client()
    try:
        response = s3_client.head_bucket(Bucket=bucket_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

#validate if folder exist inside bucket
def validate_if_folder_exist(bucket_name, folder_name):
    """Validate if a folder exist inside a bucket

    :param bucket_name: Bucket to validate
    :param folder_name: Folder to validate
    :return: True if folder exist, else False
    """
    s3_resource = create_resource()
    try:
        s3_resource.Object(bucket_name, folder_name + '/').load()
    except ClientError as e:
        logging.error(e)
        return False
    return True


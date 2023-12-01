from io import TextIOWrapper
import os
import boto3


def signin_s3(credentials: dict):
    return boto3.client('s3', 
                        aws_access_key_id     = credentials['ACCESS_KEY'],
                        aws_secret_access_key = credentials['SECRET_KEY'],)


def download_file_from_s3(local_filepath: str or TextIOWrapper, 
                          awss3_filepath: str, credentials: dict):
    
    bucket_name = credentials["BUCKET_NAME"]
    print(f"Downloading file {bucket_name}:{awss3_filepath} from AWS S3 to {local_filepath} ...")

    s3 = signin_s3(credentials)
    s3.download_file(bucket_name, awss3_filepath, local_filepath)
    print("Download file from AWS S3 successfully")


def upload_file_to_s3(local_filepath: str or TextIOWrapper, 
                      awss3_filepath: str, bucket_name: str, credentials: dict):
    
    print(f"Uploading file {local_filepath} to {bucket_name}:{awss3_filepath} in AWS S3 ...")
    s3 = signin_s3(credentials)
    s3.upload_file(local_filepath, bucket_name, awss3_filepath)
    print("Upload file to AWS S3 successfully")


def download_folder_from_s3(local_folderpath: str or TextIOWrapper, 
                            awss3_folderpath: str, credentials: dict):
    
    s3_resource = boto3.resource('s3')
    bucket = s3_resource.Bucket(credentials["BUCKET_NAME"])
    for obj in bucket.objects.filter(Prefix=awss3_folderpath):

        if obj.key == awss3_folderpath:
            continue
        else:
            path_local = obj.key.replace(awss3_folderpath, local_folderpath)
            if not os.path.exists(os.path.dirname(path_local)):
                os.makedirs(os.path.dirname(path_local))

            bucket.download_file(obj.key, path_local)  # save to same path

    print("Download folder from AWS S3 successfully")



"""
Utilities to transmit Python objects to and from an S3-compatible object storage service.
"""
from locale import Error
import pickle
from typing import Any

import boto3
import botocore
import requests

from plato.config import Config

class S3:
    def __init__(self, endpoint=None, access_key=None, secret_key=None, bucket=None):
        self.endpoint = endpoint
        self.bucket = bucket
        self.key_prefix = ""
        self.access_key = access_key
        self.secret_key = secret_key
        
        if hasattr(Config().server, 's3_endpoint_url'):
            self.endpoint = Config().server.s3_endpoint_url
            
        if hasattr(Config().server, 's3_bucket'):
            self.bucket = Config().server.s3_bucket
            
        if hasattr(Config().server, 'access_key'):
            self.access_key = Config().server.access_key
            
        if hasattr(Config().server, 'secret_key'):
            self.secret_key = Config().server.secret_key
        
        if "s3://" in self.bucket:
            bucket_part = self.bucket[5:]
            str_list = bucket_part.split("/")
            self.bucket = str_list[0]
            if len(str_list) > 1:
                self.key_prefix = bucket_part[len(self.bucket):]
        
        if os.path.exists("~/.obs/credentials") and (self.access_key == None or self.secret_key == None):
            # read from ~/.aws/credentials
            with open("~/.obs/credentials") as f:
                data = f.readlines()
                if len(data) >= 3:
                    try:
                        self.access_key = data[1].split("=")[1].replace(" ", "").replace("\n", "")
                        self.secret_key = data[2].split("=")[1].replace(" ", "").replace("\n", "")
                    except Error:
                        raise ValueError("credentials format error")
                        
        if self.endpoint == None or self.bucket == None or self.access_key == None or self.secret_key == None:
            raise ValueError("S3 does not existed")
        
        self.s3_client = boto3.client('s3', endpoint_url=self.endpoint, 
                                      aws_access_key_id=self.access_key,
                                      aws_secret_access_key=self.secret_key)
        # check the bucket exist or not
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
        except:
            try:
                self.s3_client.create_bucket(Bucket=self.bucket)
            except botocore.exception.ClientError:    
                raise ValueError("Fail to create a bucket")
    
    def send_to_s3(self, object_key, object_to_send) -> str:
        """ Sends an object to an S3-compatible object storage service.

            All S3-related credentials, such as the access key and the secret key,
            are assumed to be stored in ~/.aws/credentials by using the 'aws configure'
            command.
    
            Returns: A presigned URL for use later to retrieve the data.
        """
        try:
            # Does the object key exist already in S3?
            self.s3_client.head_object(Bucket=self.bucket, Key=object_key)
        except botocore.exceptions.ClientError:
            try:
                # Only send the object if the key does not exist yet
                data = pickle.dumps(object_to_send)
                put_url = self.s3_client.generate_presigned_url(
                        ClientMethod='put_object',
                        Params={'Bucket': self.bucket, 'Key': object_key},
                        ExpiresIn=300)
                response = requests.put(put_url, data=data)

                if response.status_code != 200:
                    raise ValueError('Error occurred sending data: status code = {}'.format(
                            response.status_code)) from None

            except botocore.exceptions.ClientError as error:
                raise ValueError('Error occurred sending data to S3: {}'.format(error)) from error

            except botocore.exceptions.ParamValidationError as error:
                raise ValueError('Incorrect parameters: {}'.format(error)) from error

    def receive_from_s3(self, object_key) -> Any:
        """ Retrieves an object from an S3-compatible object storage service.

            All S3-related credentials, such as the access key and the secret key,
            are assumed to be stored in ~/.aws/credentials by using the 'aws configure'
            command.

            Returns: The object to be retrieved.
        """
        get_url = self.s3_client.generate_presigned_url(
            ClientMethod='get_object',
            Params={'Bucket': self.bucket, 'Key': object_key},
            ExpiresIn=300)
        response = requests.get(get_url)
    
        if response.status_code == 200:
            return pickle.loads(response.content)
        else:
            raise ValueError('Error occurred sending data: request status code = {}'.format(
                response.status_code))
    
    def delete_from_s3(self, object_key):
        response = self.s3_client.delete_object(
            Bucket=self.bucket, Key=object_key)


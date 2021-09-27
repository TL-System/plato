"""
Utilities to transmit Python objects to and from an S3-compatible object storage service.
"""
import pickle
from typing import Any

import boto3
import botocore
import requests

from plato.config import Config


class S3:
    """ Manages the utilities to transmit Python objects to and from an S3-compatibile
        object storage service.
    """
    def __init__(self,
                 endpoint=None,
                 access_key=None,
                 secret_key=None,
                 bucket=None):
        """ All S3-related credentials, such as the access key and the secret key,
            are either to be stored in ~/.aws/credentials by using the 'aws configure'
            command, passed into the constructor as parameters, or specified in the
            `server` section of the configuration file.
        """
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

        if self.bucket is None:
            raise ValueError(
                "The S3 storage service has not been properly configured.")

        if "s3://" in self.bucket:
            bucket_part = self.bucket[5:]
            str_list = bucket_part.split("/")
            self.bucket = str_list[0]
            if len(str_list) > 1:
                self.key_prefix = bucket_part[len(self.bucket):]

        if self.access_key is not None and self.secret_key is not None:
            self.s3_client = boto3.client(
                's3',
                endpoint_url=self.endpoint,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key)
        else:
            # the access key and secret key are stored locally in ~/.aws/credentials
            self.s3_client = boto3.client('s3', endpoint_url=self.endpoint)

        # Does the bucket exist?
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
        except botocore.exception.ClientError:
            try:
                self.s3_client.create_bucket(Bucket=self.bucket)
            except botocore.exception.ClientError as s3_exception:
                raise ValueError("Fail to create a bucket.") from s3_exception

    def send_to_s3(self, object_key, object_to_send) -> str:
        """ Sends an object to an S3-compatible object storage service.

            Returns: A presigned URL for use later to retrieve the data.
        """
        object_key = self.key_prefix + "/" + object_key
        try:
            # Does the object key exist already in S3?
            self.s3_client.head_object(Bucket=self.bucket, Key=object_key)
        except botocore.exceptions.ClientError:
            try:
                # Only send the object if the key does not exist yet
                data = pickle.dumps(object_to_send)
                put_url = self.s3_client.generate_presigned_url(
                    ClientMethod='put_object',
                    Params={
                        'Bucket': self.bucket,
                        'Key': object_key
                    },
                    ExpiresIn=300)
                response = requests.put(put_url, data=data)

                if response.status_code != 200:
                    raise ValueError(
                        'Error occurred sending data: status code = {}'.format(
                            response.status_code)) from None

            except botocore.exceptions.ClientError as error:
                raise ValueError(
                    'Error occurred sending data to S3: {}'.format(
                        error)) from error

            except botocore.exceptions.ParamValidationError as error:
                raise ValueError(
                    'Incorrect parameters: {}'.format(error)) from error

    def receive_from_s3(self, object_key) -> Any:
        """ Retrieves an object from an S3-compatible object storage service.

            All S3-related credentials, such as the access key and the secret key,
            are assumed to be stored in ~/.aws/credentials by using the 'aws configure'
            command.

            Returns: The object to be retrieved.
        """
        object_key = self.key_prefix + "/" + object_key
        get_url = self.s3_client.generate_presigned_url(
            ClientMethod='get_object',
            Params={
                'Bucket': self.bucket,
                'Key': object_key
            },
            ExpiresIn=300)
        response = requests.get(get_url)

        if response.status_code == 200:
            return pickle.loads(response.content)

        raise ValueError(
            'Error occurred sending data: request status code = {}'.format(
                response.status_code))

    def delete_from_s3(self, object_key):
        """ Deletes an object using its key from S3. """
        __ = self.s3_client.delete_object(Bucket=self.bucket, Key=object_key)

    def lists(self):
        """ Retrieves keys to all the objects in the S3 bucket. """
        response = self.s3_client.list_objects_v2(Bucket=self.bucket)
        keys = []
        for obj in response['Contents']:
            keys.append(obj['Key'])
        return keys

"""
Utilities to transmit Python objects to and from an S3-compatible object storage service.
"""

import pickle
from typing import Any

import boto3
import botocore
import requests

from plato.config import Config


def send_to_s3(object_key, object_to_send) -> str:
    """ Sends an object to an S3-compatible object storage service.

        All S3-related credentials, such as the access key and the secret key,
        are assumed to be stored in ~/.aws/credentials by using the 'aws configure'
        command.

        Returns: A presigned URL for use later to retrieve the data.
    """
    if hasattr(Config().server,
                   's3_endpoint_url') and hasattr(Config().server, 's3_bucket'):
        s3_client = boto3.client('s3', endpoint_url=Config().server.s3_endpoint_url)

        try:
            # Does the object key exist already in S3?
            s3_client.head_object(Bucket=Config().server.s3_bucket, Key=object_key)
        except botocore.exceptions.ClientError:
            try:
                # Only send the object if the key does not exist yet
                put_url = s3_client.generate_presigned_url(
                        ClientMethod='put_object',
                        Params={'Bucket': Config().server.s3_bucket, 'Key': object_key},
                        ExpiresIn=300)
                data = pickle.dumps(object_to_send)
                response = requests.put(put_url, data=data)

                if response.status_code != 200:
                    raise ValueError('Error occurred sending data: status code = {}').format(
                        response.status_code) from None

            except botocore.exceptions.ClientError as error:
                raise ValueError('Error occurred sending data to S3: {}'.format(error)) from error

            except botocore.exceptions.ParamValidationError as error:
                raise ValueError('Incorrect parameters: {}'.format(error)) from error

        get_url = s3_client.generate_presigned_url(
            ClientMethod='get_object',
            Params={'Bucket': Config().server.s3_bucket, 'Key': object_key},
            ExpiresIn=300)

        return get_url

    raise ValueError('s3_endpoint_url and s3_bucket are not found in the configuration.')

def receive_from_s3(presigned_url) -> Any:
    """ Retrieves an object from an S3-compatible object storage service.

        All S3-related credentials, such as the access key and the secret key,
        are assumed to be stored in ~/.aws/credentials by using the 'aws configure'
        command.

        Returns: The object to be retrieved.
    """
    response = requests.get(presigned_url)
    if response.status_code == 200:
        return pickle.loads(response.content)
    else:
        raise ValueError('Error occurred sending data: request status code = {}').format(
            response.status_code)

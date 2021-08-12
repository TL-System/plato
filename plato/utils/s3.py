"""
Utilities to transmit Python objects to and from an S3-compatible object storage service.
"""

import pickle
import uuid
from typing import Tuple

import boto3
import requests

from plato.config import Config


def send_to_s3(object_to_send) -> Tuple[bool, str]:
    """ Sends an object to an S3-compatible object storage service.

        All S3-related credentials, such as the access key and the secret key,
        are assumed to be stored in ~/.aws/credentials by using the 'aws configure'
        command.

        Returns: whether the operation was successfully performed, and if so,
        a presigned URL for later retrieval.
    """
    if hasattr(Config().server,
                   'endpoint_url') and hasattr(Config().server, 'bucket'):
        s3_client = boto3.client('s3', endpoint_url=Config().server.endpoint_url)

        unique_key = uuid.uuid4().hex[:6].upper()
        url = s3_client.generate_presigned_url(
                ClientMethod='put_object',
                Params={'Bucket': Config().server.bucket, 'Key': unique_key},
                ExpiresIn=300)

        data = pickle.dumps(object_to_send)
        response = requests.put(url, data=data)
        if response.status_code == 200:
            return True, url

    return False, None

"""
Utilities to transmit Python objects to and from an S3-compatible object storage service.
"""

import pickle
from typing import Any

import boto3
import requests
from botocore.config import Config as ClientConfig
from botocore.exceptions import ClientError, ParamValidationError
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from plato.config import Config


class S3:
    """Manages the utilities to transmit Python objects to and from an S3-compatibile
    object storage service.
    """

    NETWORK_TIMEOUT = (5.0, 30.0)  # connect and read inactivity, in seconds
    MAX_ATTEMPTS = 3

    def __init__(
        self,
        endpoint: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        bucket: str | None = None,
    ):
        """All S3-related credentials, such as the access key and the secret key,
        are either to be stored in ~/.aws/credentials by using the 'aws configure'
        command, passed into the constructor as parameters, or specified in the
        `server` section of the configuration file.
        """
        self.endpoint = endpoint
        self.bucket = bucket
        self.key_prefix = ""
        self.access_key = access_key
        self.secret_key = secret_key

        if hasattr(Config().server, "s3_endpoint_url"):
            self.endpoint = Config().server.s3_endpoint_url

        if hasattr(Config().server, "s3_bucket"):
            self.bucket = Config().server.s3_bucket

        if hasattr(Config().server, "access_key"):
            self.access_key = Config().server.access_key

        if hasattr(Config().server, "secret_key"):
            self.secret_key = Config().server.secret_key

        if self.bucket is None:
            raise ValueError("The S3 storage service has not been properly configured.")

        if self.bucket.startswith("s3://"):
            bucket_part = self.bucket[5:]
            str_list = bucket_part.split("/")
            self.bucket = str_list[0]
            if len(str_list) > 1:
                self.key_prefix = bucket_part[len(self.bucket) :].strip("/")

        client_config = ClientConfig(
            connect_timeout=self.NETWORK_TIMEOUT[0],
            read_timeout=self.NETWORK_TIMEOUT[1],
            retries={"mode": "standard", "total_max_attempts": self.MAX_ATTEMPTS},
        )

        if self.access_key is not None and self.secret_key is not None:
            self.s3_client = boto3.client(
                "s3",
                endpoint_url=self.endpoint,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                config=client_config,
            )
        else:
            # the access key and secret key are stored locally in ~/.aws/credentials
            self.s3_client = boto3.client(
                "s3", endpoint_url=self.endpoint, config=client_config
            )

        # Does the bucket exist?
        try:
            self.s3_client.head_bucket(Bucket=self.bucket)
        except ClientError as error:
            if not self._is_missing(error):
                raise
            try:
                self.s3_client.create_bucket(Bucket=self.bucket)
            except ClientError as s3_exception:
                raise ValueError("Fail to create a bucket.") from s3_exception

    @staticmethod
    def _is_missing(error: ClientError) -> bool:
        return error.response.get("Error", {}).get("Code") in {
            "404",
            "NoSuchKey",
            "NoSuchBucket",
            "NotFound",
        }

    def _object_key(self, object_key: str) -> str:
        """Accept logical keys and keys returned by lists(), prefixing once."""
        if not self.key_prefix:
            return object_key
        namespace = self.key_prefix + "/"
        return (
            object_key if object_key.startswith(namespace) else namespace + object_key
        )

    def _request(self, method: str, url: str, **kwargs) -> tuple[int, bytes]:
        """Perform a presigned request with finite inactivity limits and attempts."""
        retry = Retry(
            total=self.MAX_ATTEMPTS - 1,
            backoff_factor=0.1,
            allowed_methods={"GET", "PUT"},
            status_forcelist={500, 502, 503, 504},
            respect_retry_after_header=False,
            raise_on_status=False,
        )
        with requests.Session() as session:
            adapter = HTTPAdapter(max_retries=retry)
            session.mount("http://", adapter)
            session.mount("https://", adapter)
            with session.request(
                method, url, timeout=self.NETWORK_TIMEOUT, **kwargs
            ) as response:
                return response.status_code, response.content

    def send_to_s3(self, object_key, object_to_send) -> None:
        """Sends an object to an S3-compatible object storage service if the key is unused."""
        key = self._object_key(object_key)
        try:
            # Does the object key exist already in S3?
            self.s3_client.head_object(Bucket=self.bucket, Key=key)
        except ClientError as error:
            if not self._is_missing(error):
                raise
            self.put_to_s3(object_key, object_to_send)

    def receive_from_s3(self, object_key) -> Any:
        """Retrieves an object from an S3-compatible object storage service.

        All S3-related credentials, such as the access key and the secret key,
        are assumed to be stored in ~/.aws/credentials by using the 'aws configure'
        command.

        Returns: The object to be retrieved.
        """
        object_key = self._object_key(object_key)
        get_url = self.s3_client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": self.bucket, "Key": object_key},
            ExpiresIn=300,
        )
        status_code, content = self._request("GET", get_url)

        if status_code == 200:
            # Objects are pickles from trusted participants/storage writers only.
            return pickle.loads(content)
        if status_code == 404:
            raise FileNotFoundError(f"S3 object '{object_key}' does not exist.")

        raise ValueError(
            f"Error occurred receiving data: request status code = {status_code}"
        )

    def put_to_s3(self, object_key, object_to_put) -> None:
        """Uploads an object regardless of whether the key already exists."""
        object_key = self._object_key(object_key)
        try:
            data = pickle.dumps(object_to_put)
            put_url = self.s3_client.generate_presigned_url(
                ClientMethod="put_object",
                Params={"Bucket": self.bucket, "Key": object_key},
                ExpiresIn=300,
            )
            status_code, _content = self._request("PUT", put_url, data=data)

            if status_code != 200:
                raise ValueError(
                    f"Error occurred sending data: status code = {status_code}"
                ) from None

        except ClientError as error:
            raise ValueError(f"Error occurred sending data to S3: {error}") from error

        except ParamValidationError as error:
            raise ValueError(f"Incorrect parameters: {error}") from error

    def delete_from_s3(self, object_key):
        """Deletes an object using its key from S3."""
        self.s3_client.delete_object(
            Bucket=self.bucket, Key=self._object_key(object_key)
        )

    def lists(self):
        """Retrieve all physical object keys in the configured namespace."""
        prefix = self.key_prefix + "/" if self.key_prefix else ""
        pages = self.s3_client.get_paginator("list_objects_v2").paginate(
            Bucket=self.bucket, Prefix=prefix
        )
        keys = []
        for page in pages:
            keys.extend(obj["Key"] for obj in page.get("Contents", []))
        return keys

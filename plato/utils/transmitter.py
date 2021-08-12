import minio
from urllib.parse import urlparse
from minio.error import S3Error
import pickle
import os

class S3Transmitter():

    def __init__(self, endpoint, access_key, secret_key, bucket, tmp="./tmp.data"):
        self.endpoint = endpoint
        self.bucket = bucket
        self.access_key = access_key
        self.secret_key = secret_key
        self.tmp = tmp
        self.url = urlparse(endpoint)
        self.use_ssl = self.url.scheme == 'https' if self.url.scheme else True
        print(self.use_ssl, self.url.scheme, self.url.netloc)
        self.minio = minio.Minio(self.url.netloc, access_key, secret_key, secure=self.use_ssl)
    
    def recv(self, key):
        try:
            self.minio.fget_object(self.bucket, key, self.tmp)
        except S3Error as exc:
            print("error occurred during download.", exc)
            if os.path.exists(self.tmp):
                os.remove(self.tmp)
        data = pickle.load(open(self.tmp, "rb"))
        if os.path.exists(self.tmp):
            os.remove(self.tmp)
        return data
    
    def send(self, key, data):
        pickle.dump(data, open(self.tmp, "wb"))
        try:
            self.minio.fput_object(self.bucket, key, self.tmp)
        except S3Error as exc:
            print("error occurred during send.", exc)
            if os.path.exists(self.tmp):
                os.remove(self.tmp)
        if os.path.exists(self.tmp):
            os.remove(self.tmp)
        
    def recvFile(self, key, path):
        try:
            self.minio.fget_object(self.bucket, key, path)
        except S3Error as exc:
            print("error occurred during download.", exc)
        
    def sendFile(self, key, path):
        try:
            self.minio.fput_object(self.bucket, key, path)
        except S3Error as exc:
            print("error occurred during send.", exc)
            
    def delete(self, key):
        try:
            self.minio.remove_object(self.bucket, key)
        except S3Error as exc:
            print("error occurred during delete.", exc)
            
    def compress(self):  # for compressing weights, feature data, data after distillation
        pass

    def decompress(self):
        pass

if __name__ == "__main__":
    # obs://
    s3 = S3Transmitter(endpoint, access_key, secret_key, bucket)
    s3.sendFile("weights", "./1.txt")
    s3.recvFile("weights", "./2.txt")
    s3.delete("weights")
    x = list(range(100))
    print(x)
    s3.send("test", x)
    y = s3.recv("test")
    print(y)
    s3.delete("test")
    
import logging
import boto3
from botocore.exceptions import ClientError
import time

def upload_file(file_name, bucket, upload_path):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, upload_path, ExtraArgs={'ACL':'public-read'})
    except ClientError as e:
        logging.error(e)
        return False
    return True

# while True:
#     upload_file(file_name="overnight_v4_log.txt", bucket="3co")
#     upload_file(file_name="6d_calibration_v7_90_degrees_hardline.tsv", bucket="3co")
#     time.sleep(60.0)
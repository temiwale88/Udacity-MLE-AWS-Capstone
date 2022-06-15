
import base64
import logging
import json
import boto3
import pickle
import os
# import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

print('Loading Lambda function')

runtime=boto3.Session().client('sagemaker-runtime')
# endpoint_Name='pytorch-inference-2022-06-13-17-34-37-720'
endpoint_Name = os.environ['ENDPOINT_NAME']
bucket = os.environ['BUCKET']
s3 = boto3.resource('s3')
# bucket = 'sagemaker-us-east-1-328945632120'
pkl_key = "dogImages/classes/dog_breeds_labels.pkl"
classes = pickle.loads(s3.Bucket(bucket).Object(pkl_key).get()['Body'].read())

def lambda_handler(event, context):

    #x=event['content']
    #aa=x.encode('ascii')
    #bs=base64.b64decode(aa)
    # print('Context:::',context)
    # print('EventType::',type(event))
    bs=event
    runtime=boto3.Session().client('sagemaker-runtime')
    
    response=runtime.invoke_endpoint(EndpointName=endpoint_Name,
                                    ContentType="application/json",
                                    Accept='application/json',
                                    #Body=bytearray(x)
                                    Body=json.dumps(bs))
    
    result=response['Body'].read().decode('utf-8')
    sss=json.loads(result)[0]
    # max_value = max(sss) #let's retrieve highest number
    
    # -- Another way - more dynamic for multiple possibilities e.g. 1st, 2nd, 3rd etc. predicted dogs -- #
    list1 = list(set(sss))
    list1.sort()
    max_value = list1[-1]
    second_max_value = list1[-2] #let's retrieve second to highest number
    max_index = sss.index(max_value) #let's retrieve the index of the max number
    second_max_index = sss.index(second_max_value)
    predicted_dog = classes[max_index]
    second_predicted_dog = classes[second_max_index]
    final_response = {"first_confidence": max_value, "first_predicted_dog": predicted_dog, "second_confidence": second_max_value, "second_predicted_dog": second_predicted_dog}
    # final_response = {"first_confidence": max_value}
    return {
        'statusCode': 200,
        'headers' : { 'Content-Type' : 'text/plain', 'Access-Control-Allow-Origin' : '*' },
        'type-result':str(type(result)),
        'Content-Type-In':str(context),
        'body' : json.dumps(final_response)
        #'updated_result':str(updated_result)

        }
    
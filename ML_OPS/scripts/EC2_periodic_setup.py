import boto3
from botocore.exceptions import ClientError
import time
from EC2_one_time_setup import wait_for_status
def stop_instances(instance_id, ec2):
    print("EC2 Instance Stop")
    ec2.stop_instances(InstanceIds=instance_id)

    wait_for_status(instance_id, 'stopped')

#stop_instances([instance_id])

def start_instances(instance_id, ec2):
    print("EC2 Instance Start")
    ec2.start_instances(InstanceIds=instance_id)

    wait_for_status(ec2, instance_id, 'running')
    # after instance is started, a new public ip is assinged
    response = ec2.describe_instances(InstanceIds=instance_id)
    public_ip = response['Reservations'][0]['Instances'][0]['PublicIpAddress']
    return public_ip
#start_instances([instance_id])

def stop_instances(instance_id, ec2):
    print("EC2 Instance Stop")
    ec2.stop_instances(InstanceIds=instance_id)

    wait_for_status(ec2,instance_id, 'stopped')
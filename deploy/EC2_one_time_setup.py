import boto3
from botocore.exceptions import ClientError
import time

instance_name = "mlops-prod"
ec2 = boto3.client('ec2')

def check_if_instance_exists(instance_name, ec2):
    response = ec2.describe_instances()
    instance_id = ""

    for resp in response['Reservations']:
        resp = resp['Instances'][0]
        tags = resp.get('Tags', [])

        for tag in tags:
            if tag.get("Key", "") == "Name" and tag.get("Value", "") == instance_name:
                instance_id = resp['InstanceId']

    if instance_id == "":
        print(f"No instance found with name {instance_name}, creating one.")
        instance_id = create_ec2_instance(instance_name, ec2)
        print("instance created")
        # raise("Stop here!!!")
    else:
        print("Instance already existed.")
    print(f"Instance ID: {instance_id}")
    return instance_id

def create_ec2_instance(instance_name, ec2):
    response = ec2.run_instances(
        ImageId='ami-0197c13a4f68c9360',
        MinCount=1,
        MaxCount=1,
        InstanceType='t2.large', # t2.micro also possible
        KeyName='jani2',   # look at explanation notebook for creating a key
        BlockDeviceMappings=[
            {
                "DeviceName": "/dev/xvda",
                'Ebs': {
                    'DeleteOnTermination': True,
                    'VolumeSize': 120
                }
            }
        ]

    )
    instance_id = response['Instances'][0]['InstanceId']

    ec2.create_tags(Resources=[instance_id], Tags=[
        {
            'Key': 'Name',
            'Value': instance_name
        }
    ])
    return instance_id


def create_security_group(group_name, ec2):
    response = ec2.describe_security_groups()

    # check if group already exist
    security_group_id = [x['GroupId'] for x in response['SecurityGroups'] if x['GroupName'] == group_name]

    if security_group_id == []:
        response = ec2.create_security_group(
            GroupName=group_name,
            Description="Security group for testing"
        )
        security_group_id = response['GroupId']
    else:
        security_group_id = security_group_id[0]

    return security_group_id

def update_security_group(group_id, protocol, port, cidr):
    """Updates a security group with a new ingress rule.

        Args:
            group_id (str): The ID of the security group to update.
            protocol (str): The protocol for the new rule (e.g., 'tcp', 'udp').
            port (int): The port number for the new rule.
            cidr (str): The CIDR block (allowed IP-adresses) for the new rule.

        Raises:
            ClientError: If an error occurs during the API call.
    """
    try:
        response = ec2.authorize_security_group_ingress(
            GroupId = group_id,
            IpPermissions=[
                {
                    'IpProtocol': protocol,
                    'FromPort': port,
                    'ToPort': port,
                    'IpRanges': [{'CidrIp': cidr}]
                }
            ]
        )
    except ClientError as e:
        if e.response['Error']['Code']=='InvalidPermission.Duplicate':
            print('This rule is already there')
        else:
            print("an error as occured!")
            print(e)

# # port 22 for SSH
# update_security_group(security_group_id,
#                       'tcp',
#                       22,
#                       '0.0.0.0/0')
# # port 80 for html
# update_security_group(security_group_id,
#                       'tcp',
#                       80,
#                       '0.0.0.0/0')
# # port 8501 for streamlit access
# update_security_group(security_group_id,
#                       'tcp',
#                       8501,
#                       '0.0.0.0/0')
#
# update_security_group(security_group_id,
#                       'tcp',
#                       8502,
#                       '0.0.0.0/0')
# # updating all changes on the group
# ec2.modify_instance_attribute(InstanceId=instance_id, Groups=[security_group_id])

def wait_for_status(instance_id, target_status):
    while True:
        response = ec2.describe_instances(InstanceIds=instance_id)

        status = response['Reservations'][0]['Instances'][0]['State']['Name']

        if status == target_status:
            print("Instance is in {} state".format(target_status))
            break

        time.sleep(10)


def terminate_instances(instance_id):
    print("EC2 Instance Termination")
    ec2.terminate_instances(InstanceIds=instance_id)

    wait_for_status(instance_id, 'terminated')

# # describe IAM role
# iam = boto3.client('iam')
#
# response = iam.get_role(RoleName=role_name)
#
# role_arn = response['Role']['Arn']
#
# # Ensure there is an instance profile with the same name as the role
# instance_profile_name = role_name
# try:
#     iam.get_instance_profile(InstanceProfileName=instance_profile_name)
# except iam.exceptions.NoSuchEntityException:
#     # Create an instance profile if it doesn't exist
#     iam.create_instance_profile(InstanceProfileName=instance_profile_name)
#     # Add role to the instance profile
#     iam.add_role_to_instance_profile(
#         InstanceProfileName=instance_profile_name,
#         RoleName=role_name
#     )
#
# # Attach the instance profile to the EC2 instance
# ec2.associate_iam_instance_profile(
#     IamInstanceProfile={
#         'Name': instance_profile_name
#     },
#     InstanceId=instance_id
# )
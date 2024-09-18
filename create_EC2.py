from deploy.EC2_periodic_setup import (wait_for_status,
                                       stop_instances,
                                       start_instances)

from deploy.EC2_one_time_setup import (check_if_instance_exists,
                                       create_security_group,
                                       create_ec2_instance,
                                       update_security_group,
                                       terminate_instances)
import boto3

instance_name = "mlops-prod"
ec2 = boto3.client('ec2')

instance_id = check_if_instance_exists(instance_name, ec2)

if not instance_id:
    instance_id = create_ec2_instance(instance_name, ec2)

security_group_name = "group1"

security_group_id = create_security_group(security_group_name, ec2)


# port 22 for SSH
update_security_group(security_group_id,
                      'tcp',
                      22,
                      '0.0.0.0/0')
# port 80 for html
update_security_group(security_group_id,
                      'tcp',
                      80,
                      '0.0.0.0/0')
# port 8501 for streamlit access
update_security_group(security_group_id,
                      'tcp',
                      8501,
                      '0.0.0.0/0')

update_security_group(security_group_id,
                      'tcp',
                      8502,
                      '0.0.0.0/0')

# updating all changes on the group
ec2.modify_instance_attribute(InstanceId=instance_id,
                              Groups=[security_group_id])

start_instances([instance_id])

# after instance is started, a new public ip
# is assinged
response = ec2.describe_instances(InstanceIds=[instance_id])
public_ip = response['Reservations'][0]['Instances'][0]['PublicIpAddress']
print(public_ip)

#stop_instances([instance_id])
#terminate_instances(instance_id)



from fabric_base import *
import fabric_base
import time


AMI = 'ami-aba229cb'
SECURITY_GROUP_IDS=["sg-1542296d"]
AWS_REGION = 'us-west-2'
SPOT_PRICE=15.0
MY_AWS_KEY= 'imagenet_exps'
instance_name_base = "hyperband"
role_name = instance_name_base
CONDA_DIR = "$HOME/anaconda3"
INSTANCE_TYPE = 'p2.16xlarge'

fabric_base.PREFIX_BASE[0] = instance_name_base


@task
def setup(prefix, num_instances=1):
    ret_val = execute(launch, num_instances=num_instances, spot_price=SPOT_PRICE, my_aws_key=MY_AWS_KEY, region=AWS_REGION, ami=AMI, instance_type=INSTANCE_TYPE, name_prefix=[prefix])
    print(ret_val)
    time.sleep(30)

@task
def provision(prefix=instance_name_base):
    execute(hosts, prefix)
    execute(update_bashrc)
    try:
        execute(install_aws)
    except:
        pass
   
    try:
        execute(pywren_setup)
    except:
        pass
    execute(pictureweb_setup)




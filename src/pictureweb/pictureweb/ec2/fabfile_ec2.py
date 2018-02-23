"""
Fabric file to help with launching EC2 P2 instancesand
getting GPU support set up. Also installs latest
anaconda and then tensorflow. Use:

fab -f fabfile_awsgpu.py launch

# wait until you can ssh into the instance with
fab -f fabfile_awsgpu.py -R mygpu ssh

# install everything
fab -f fabfile_awsgpu.py -R mygpu basic_setup cuda_setup8 anaconda_setup tf_setup keras_setup torch_preroll torch_setup_solo


# when you're done, terminate
fab -f fabfile_awsgpu.py -R mygpu terminate


Took inspiration from:
https://aws.amazon.com/blogs/aws/new-p2-instance-type-for-amazon-ec2-up-to-16-gpus/

"""


from fabric.api import local, env, run, put, cd, task, \
    sudo, settings, warn_only, lcd, path, get, parallel

from fabric.operations import put

from fabric.contrib import project
import boto3
import os
import base64
import time

AMI = 'ami-9daf2cfd'
SECURITY_GROUP_IDS=["sg-1542296d"]
AWS_REGION = 'us-west-2'
SPOT_PRICE=10.0
MY_AWS_KEY= 'imagenet_exps'
instance_name_base = "big_cpu"
role_name = instance_name_base
CONDA_DIR = "$HOME/anaconda3"
INSTANCE_TYPE = 'x1.32xlarge'

DASK_WORKER_COMMAND = "dask-worker --nprocs 100 --nthreads 1 --memory-limit=auto"

def tags_to_dict(d):
    return {a['Key'] : a['Value'] for a in d}

@task
def not_hosts(prefix='', user='ubuntu'):
    hosts(prefix, user, negate=True)
@task
def hosts(prefix=instance_name_base, user='ubuntu', negate=False):
    print(prefix)
    negate=bool(negate)
    res = []
    ec2 = boto3.resource('ec2', region_name=AWS_REGION)
    for i in ec2.instances.all():
        if i.state['Name'] == 'running':
            if (i.tags == None): continue
            d = tags_to_dict(i.tags)
            if negate ^ d['Name'].startswith(prefix):
                res.append(i.public_dns_name)
    print("RUNNING COMMAND ON HOSTS:", res)
    env.hosts = res
    env.user = user




def _create_instances(num_instances,
        region,
        spot_price,
        ami,
        key_name,
        instance_type,
        block_device_mappings,
        security_group_ids,
        ebs_optimized):

    ''' Function graciously borrowed from Flintrock ec2 wrapper
        https://raw.githubusercontent.com/nchammas/flintrock/00cce5fe9d9f741f5999fddf2c7931d2cb1bdbe8/flintrock/ec2.py
    '''

    ec2 = boto3.resource(service_name='ec2', region_name=region)
    spot_requests = []
    try:
        if spot_price:
            print("Requesting {c} spot instances at a max price of ${p}...".format(
                c=num_instances, p=spot_price))
            client = ec2.meta.client
            spot_requests = client.request_spot_instances(
                SpotPrice=str(spot_price),
                InstanceCount=num_instances,
                LaunchSpecification={
                    'ImageId': ami,
                    'KeyName': key_name,
                    'InstanceType': instance_type,
                    'BlockDeviceMappings': block_device_mappings,
                    'SecurityGroupIds': security_group_ids,
                    'EbsOptimized': ebs_optimized})['SpotInstanceRequests']

            request_ids = [r['SpotInstanceRequestId'] for r in spot_requests]
            pending_request_ids = request_ids

            while pending_request_ids:
                print("{grant} of {req} instances granted. Waiting...".format(
                    grant=num_instances - len(pending_request_ids),
                    req=num_instances))
                time.sleep(30)
                spot_requests = client.describe_spot_instance_requests(
                    SpotInstanceRequestIds=request_ids)['SpotInstanceRequests']

                failed_requests = [r for r in spot_requests if r['State'] == 'failed']
                if failed_requests:
                    failure_reasons = {r['Status']['Code'] for r in failed_requests}
                    raise Exception(
                        "The spot request failed for the following reason{s}: {reasons}"
                        .format(
                            s='' if len(failure_reasons) == 1 else 's',
                            reasons=', '.join(failure_reasons)))

                pending_request_ids = [
                    r['SpotInstanceRequestId'] for r in spot_requests
                    if r['State'] == 'open']

            print("All {c} instances granted.".format(c=num_instances))

            cluster_instances = list(
                ec2.instances.filter(
                    Filters=[
                        {'Name': 'instance-id', 'Values': [r['InstanceId'] for r in spot_requests]}
                    ]))
        else:
            # Move this to flintrock.py?
            print("Launching {c} instance{s}...".format(
                c=num_instances,
                s='' if num_instances == 1 else 's'))

            # TODO: If an exception is raised in here, some instances may be
            #       left stranded.
            cluster_instances = ec2.create_instances(
                MinCount=num_instances,
                MaxCount=num_instances,
                ImageId=ami,
                KeyName=key_name,
                InstanceType=instance_type,
                BlockDeviceMappings=block_device_mappings,
                SecurityGroupIds=security_group_ids,
                EbsOptimized=ebs_optimized)
        time.sleep(10)  # AWS metadata eventual consistency tax.
        return cluster_instances
    except (Exception, KeyboardInterrupt) as e:
        if not isinstance(e, KeyboardInterrupt):
            print(e)
        if spot_requests:
            request_ids = [r['SpotInstanceRequestId'] for r in spot_requests]
            if any([r['State'] != 'active' for r in spot_requests]):
                print("Canceling spot instance requests...")
                client.cancel_spot_instance_requests(
                    SpotInstanceRequestIds=request_ids)
            # Make sure we have the latest information on any launched spot instances.
            spot_requests = client.describe_spot_instance_requests(
                SpotInstanceRequestIds=request_ids)['SpotInstanceRequests']
            instance_ids = [
                r['InstanceId'] for r in spot_requests
                if 'InstanceId' in r]
            if instance_ids:
                cluster_instances = list(
                    ec2.instances.filter(
                        Filters=[
                            {'Name': 'instance-id', 'Values': instance_ids}
                        ]))
        raise Exception("Launch failure")




@task
def launch(num_instances=1, spot_price=SPOT_PRICE, my_aws_key=MY_AWS_KEY,  region=AWS_REGION, ami=AMI, security_group_ids=SECURITY_GROUP_IDS, instance_type=INSTANCE_TYPE, name_prefix=instance_name_base):
    ec2 = boto3.resource('ec2', region_name=region)
    num_instances = int(num_instances)
    BlockDeviceMappings=[
        {
            'DeviceName': '/dev/xvda',
            'Ebs': {
                'VolumeSize': 1024,
                'DeleteOnTermination': True,
                'VolumeType': 'standard',
                'SnapshotId' : 'snap-c87f35ec'
            },
        },
    ]

    instances = _create_instances(num_instances, region, spot_price, ami, my_aws_key, instance_type, BlockDeviceMappings, security_group_ids, ebs_optimized=False)


    for i,inst in enumerate(instances):

        inst.wait_until_running()
        inst.reload()
        inst.create_tags(
            Resources=[
                inst.instance_id
            ],
            Tags=[
            {
                'Key': 'Name',
                'Value': '{0}_{1}'.format(instance_name_base, i)
            },
            ]
        )


@task
def ssh():
    local("ssh -A " + env.host_string)

@task
def install_aws():
    run("PATH=~/anaconda3/bin/:$PATH pip install awscli")
    run("mkdir /home/ubuntu/.aws")
    put("~/.aws/", "/home/ubuntu/")


@task
def do(string=""):
    run(string)


@task
def start_dask_worker(master, port):
    run("PATH=~/anaconda3/bin/:$PATH tmux new -d -s dask-worker \'{0} {1}:{2}\'".format(DASK_WORKER_COMMAND, master, port))

@task
def kill_dask_worker():
    run("PATH=~/anaconda3/bin/:$PATH tmux kill-session -t dask-worker")

@task
def cuda_setup():

    run("wget http://us.download.nvidia.com/XFree86/Linux-x86_64/352.99/NVIDIA-Linux-x86_64-352.99.run")
    run("wget http://developer.download.nvidia.com/compute/cuda/7.5/Prod/local_installers/cuda_7.5.18_linux.run")
    run("chmod +x NVIDIA-Linux-x87_64-352.99.run")
    run("chmod +x cuda_7.5.18_linux.run")
    sudo("./NVIDIA-Linux-x86_64-352.99.run --silent") # still requires a few prompts
    sudo("./cuda_7.5.18_linux.run --silent --toolkit --samples")   # Don't install driver, just install CUDA and sample
    #
    sudo("nvidia-smi -pm 1")
    sudo("nvidia-smi -acp 0")
    sudo("nvidia-smi --auto-boost-permission=0")
    sudo("nvidia-smi -ac 2505,875")

    # cudnn
    with cd("/usr/local"):
        sudo("wget http://people.eecs.berkeley.edu/~jonas/cudnn-8.0-linux-x64-v5.1.tgz")
        sudo("tar xvf cudnn-8.0-linux-x64-v5.1.tgz")

    sudo('echo "/usr/local/cuda/lib64/" >> /etc/ld.so.conf')
    sudo('echo "/usr/local/cuda/extras/CPUTI/lib64/" >> /etc/ld.so.conf')
    sudo('ldconfig')

@task
def cuda_setup8():

    run("wget http://us.download.nvidia.com/XFree86/Linux-x86_64/370.28/NVIDIA-Linux-x86_64-370.28.run")
    run("wget https://developer.nvidia.com/compute/cuda/8.0/prod/local_installers/cuda_8.0.44_linux-run")
    run("mv NVIDIA-Linux-x86_64-370.28.run driver.run")
    run("mv cuda_8.0.44_linux-run cuda.run")
    run("chmod +x driver.run")
    run("chmod +x cuda.run")
    sudo("./driver.run --silent") # still requires a few prompts
    sudo("./cuda.run --silent --toolkit --samples")   # Don't install driver, just install CUDA and sample
    #
    sudo("nvidia-smi -pm 1")
    sudo("nvidia-smi -acp 0")
    sudo("nvidia-smi --auto-boost-permission=0")
    sudo("nvidia-smi -ac 2505,875")

    # cudnn
    with cd("/usr/local"):
        sudo("wget http://people.eecs.berkeley.edu/~jonas/cudnn-8.0-linux-x64-v5.1.tgz")
        sudo("tar xvf cudnn-8.0-linux-x64-v5.1.tgz")

    sudo('echo "/usr/local/cuda/lib64/" >> /etc/ld.so.conf')
    sudo('echo "/usr/local/cuda/extras/CPUTI/lib64/" >> /etc/ld.so.conf')
    sudo('ldconfig')

@task
def anaconda_setup():
    run("wget https://repo.continuum.io/archive/Anaconda2-4.2.0-Linux-x86_64.sh")
    run("chmod +x Anaconda2-4.2.0-Linux-x86_64.sh")
    run("./Anaconda2-4.2.0-Linux-x86_64.sh -b -p {}".format(CONDA_DIR))
    run('echo "export PATH={}/bin:$PATH" >> .bash_profile'.format(CONDA_DIR))
    run("conda upgrade -q -y --all")
    run("conda install -q -y pandas scikit-learn scikit-image matplotlib seaborn ipython")
    run("pip install ruffus glob2 awscli")

@task
def tf_setup():
    run("pip install --ignore-installed --upgrade {}".format(TF_URL))


@task
def terminate(prefix=instance_name_base):
    ec2 = boto3.resource('ec2', region_name=AWS_REGION)

    insts = []
    for i in ec2.instances.all():
        if i.state['Name'] == 'running':
            d = tags_to_dict(i.tags)
            if d['Name'].startswith(prefix):
                i.terminate()
                insts.append(i)


if __name__ == "__main__":
    hosts()



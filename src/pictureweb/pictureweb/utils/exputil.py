import gspread
from oauth2client.service_account import ServiceAccountCredentials
import datetime
import collections
import os
import hashlib
import git
import json
import boto3
import tarfile


scope = ['https://spreadsheets.google.com/feeds']
credentials = ServiceAccountCredentials.from_json_keyfile_name('/tmp/gspread.json', scope)
gc = gspread.authorize(credentials)

# Hard coded
raw_sheet = gc.open("Imagenet DeathMarch 2017").worksheet('raw')


def flatten(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def column_name_to_dict_key(name):
    return name.replace(" ", "_")

def dict_name_to_column_key(name):
    return name.replace("_", " ")

def update_sheet_add_row(result_config, sheet):
    col_labels = list(map(column_name_to_dict_key, filter(lambda x: len(x), raw_sheet.row_values(1))))
    result_row = ["N/A" for i in range(len(col_labels))]
    result_config = flatten(result_config)
    for i, label in enumerate(col_labels):
        value = result_config.get(label)
        if (value != None):
            result_row[i] = value
    sheet.append_row(result_row)

def overwrite_sheet(result_configs, sheet=raw_sheet):
    while(sheet.row_count > 1):
        sheet.delete_row(sheet.row_count)
    for config in result_configs:
        update_sheet_add_row(config, sheet)

def clear_sheet(sheet=raw_sheet):
    overwrite_sheet([], sheet)

def overwrite_sheet_from_s3(sheet=raw_sheet, filter_fn= lambda x: True, bucket="pictureweb2017experiments"):
    clear_sheet(sheet)
    s3 = boto3.client('s3')
    # Get objects, run through filter function, blah blah 
    keys = []
    s3_response = s3.list_objects_v2(Bucket=bucket)
    for obj in s3_response["Contents"]:
        keys.append(obj["Key"])

    while(s3_response["IsTruncated"]):
        for obj in s3_response["Contents"]:
            keys.append(obj["Key"])
        s3_response = s3.list_objects_v2(Bucket='pictureweb2017experiments', ContinuationToken=s3_reponse['NextContinuationToken'])

    configs = []
    keys = list(filter(lambda x: x.endswith(".json"), keys))
    for key in keys:
        print(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
        config = json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())
        if (filter_fn(config)):
            configs.append(config)
    overwrite_sheet(configs, sheet)



def find_depth(path):
    if (path[-1] == "/"):
        path = path[:-1]
    parent, child = os.path.split(path)
    depth = 0
    while(parent != "/"):
        print(parent, child)
        parent, child = os.path.split(parent)
        depth += 1
    return depth


def create_code_tarball_and_hash(repo_path):
    now = datetime.datetime.now().__str__().replace(" ", "-")
    tar_path = "./picture_web_{0}.tar".format(now)
    if (repo_path[-1] == "/"):
        repo_path = repo_path[:-1]
    parent, reponame = os.path.split(repo_path)

    cmd = "/bin/bash -c 'cd {1} ; tar -cf {0} {2} --exclude-from <(find {2} -size +3M)'".format(tar_path, parent, reponame)
    print(cmd)
    os.system(cmd)
    tar_path = "{0}/{1}".format(parent, tar_path)
    md5_hash = hashlib.md5(open(tar_path, 'rb').read()).hexdigest()
    return (md5_hash, tar_path)

def create_results_config(solve_params, feat_params, results, tarball_hash, notes, **kwargs):
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    result_config = {}
    result_config["solve_params"] = solve_params
    result_config["feat_params"] = feat_params
    result_config["results"] = results
    result_config["git_commit_hash"] = sha
    result_config["notes"] = notes
    for k,v in kwargs.items():
        result_config[key] = v
    m = hashlib.md5()
    m.update(tarball_hash.encode('utf-8'))
    m.update(str(result_config).encode('utf-8'))
    result_config["experiment_hash"] = m.hexdigest()
    result_config["timestamp"] = datetime.datetime.now().__str__()
    return result_config

def save_results(solve_params, feat_params, results, notes, repo_path="/data/vaishaal/pictureweb", sheet=raw_sheet, update_sheet=True, **kwargs):
    tarball_hash, tar_path = create_code_tarball_and_hash(repo_path)
    result_config = create_results_config(solve_params, feat_params, results, tarball_hash, notes, **kwargs)
    upload_results_to_s3(tar_path, result_config)
    print(result_config["experiment_hash"])
    if (update_sheet):
        update_sheet_add_row(result_config, sheet)

def upload_results_to_s3(tarball_path, result_config, bucket="pictureweb2017experiments"):
    s3 = boto3.client('s3')
    key_base = result_config["experiment_hash"]
    json_key = key_base + ".json"
    tar_key = key_base + ".tar"
    json_str = json.dumps(result_config)
    s3.put_object(Bucket=bucket, Key=json_key, Body=json_str.encode('utf-8'))
    s3.put_object(Bucket=bucket, Key=tar_key, Body=open(tarball_path, "rb"))

def snapshot(repo_path="/home/ubuntu/pictureweb", bucket="picturewebsnapshots"):
    (md5_hash, tarball_path) = create_code_tarball_and_hash(repo_path)
    s3 = boto3.client('s3')
    parent, key = os.path.split(tarball_path)
    s3.put_object(Bucket=bucket, Key=key, Body=open(tarball_path, "rb"))



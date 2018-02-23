import boto3
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import sys
sys.path.insert(0, "..")
from distributed import sharded_matrix
import distributed.distributed as D 
from loaders.imagenet_load import orient
from conv import coates_ng_help

scope = ['https://spreadsheets.google.com/feeds']
credentials = ServiceAccountCredentials.from_json_keyfile_name('/tmp/gspread.json', scope)
gc = gspread.authorize(credentials)
# Get the service resource
sqs = boto3.resource('sqs')

# Get the queue
queue = sqs.get_queue_by_name(QueueName='picturewebfeaturize')

raw_sheet = gc.open("Imagenet DeathMarch 2017").worksheet('Hyperband Round 1')
num_hps = 128
row_labels = list(map(lambda x: x.replace(",", "").strip(), raw_sheet.row_values(1)[:5]))
for i in range(2, num_hps+2):
        hp_draw = dict(zip(row_labels,raw_sheet.row_values(i)[:5]))
        queue.send_message(MessageBody=json.dumps(hp_draw))

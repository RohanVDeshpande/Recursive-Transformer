import math
import random
import json
import os
import sys
import time
import datetime
from termcolor import colored

import torch
import torch.nn as nn
import argparse

from . import data
from . import wsrt

parser = argparse.ArgumentParser(description='Forced Recursive Transformer')
parser.add_argument('--model-config', type=str, required=True,
                    help='Model json config')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--params', type=str, help='Model params path', required=True)
parser.add_argument('--dict', type=str, help='Dictionary path')

args = parser.parse_args()

assert os.path.isdir("output/"), "You need a folder called 'output' in order to save test predictions"
assert os.path.isdir("tb/"), "You need a folder called 'tb' for tensorboard"
assert os.path.isdir("output/dict/"), "You need a folder called 'dict' in order to save/load a dictionary"
assert os.path.isdir("checkpoints/"), "You need a folder called 'checkpoints' in order to save model params"

# Set the random seed manually for reproducibility.
# torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
    	utils.confirm("You have a CUDA device, so you should probably run with --cuda.")
    else:
    	print("Using CUDA device")

device = torch.device("cuda" if args.cuda else "cpu")


with open(args.model_config) as f:
  model_config = json.load(f)
  dictionary_path = args.dict if args.dict else "output/dict/{}_dict.json".format(model_config["NAME"])

  with open(model_config["DATASET_CONFIG"]) as f:
  	dataset_config = json.load(f)

dataset = data.Dataset(dataset_config)
dataset.loadDictionary(dictionary_path)
dataset.device = device
dataset.dictionary.freeze_dict = True

model_config["TOKENS"] = dataset.tokens()
model_config["SRC_LEN"] = dataset.SRC_LEN
model_config["TGT_LEN"] = dataset.TGT_LEN


model = wsrt.WSRT(model_config)
model.device = device
model.to(device)

model.load_state_dict(torch.load(args.params, map_location=torch.device(device)))
model.eval()

src_text = "51-19"
src_text = dataset.START + src_text + dataset.END
src_text += (dataset.SRC_LEN - len(src_text)) * dataset.PADDING

src_indicies = dataset.text2tensor(src_text).view(-1, 1)
# print(src_indicies)
output = model.predict(src_indicies, None, dataset.dictionary.word2idx[dataset.START])
# print(output)
# print(output.shape)
output_string = dataset.tensor2text(output.view(-1))   # .split(dataset.END)[0]
print(" ", colored(output_string, "blue"))

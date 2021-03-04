import random
import math
import random
from termcolor import colored
import json
import utils
from utils import AverageMeter
import os
import sys
import time
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import frt

import data
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='Forced Recursive Transformer')
parser.add_argument('--model-config', type=str, required=True,
                    help='Model json config')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--params', type=str, required=True, help='Model params path for test')

args = parser.parse_args()

assert os.path.isdir("output/"), "You need a folder called 'output' in order to save model data / test predictions"

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
  with open(model_config["DATASET_CONFIG"]) as f:
  	dataset_config = json.load(f)




dataset = data.Dataset(dataset_config)
dataset.device = device

def make_q_a(num1, num2):
	add = (random.random() >= 0.5)
	q = "{}{}{}".format(num1, "+" if add else "-", num2)
	a = "{}".format(num1 + num2 if add else num1 - num2)
	return "{}\t{}\t1\tsuper_simple_add_sub".format(q, a)

def sample_q_a():
	return make_q_a(random.randint(0, 4), random.randint(0, 4))

dataset.BATCH_SIZE = 1

dataset.buildDataset([sample_q_a() for i in range(dataset.BATCH_SIZE)])

dataloader = DataLoader(dataset, batch_size=dataset_config["BATCH_SIZE"], shuffle=dataset_config["SHUFFLE"], num_workers=dataset_config["WORKERS"],
           pin_memory=dataset_config["PIN_MEMORY"], prefetch_factor=dataset_config["PREFETCH_FACTOR"],
           persistent_workers=True, collate_fn=data.dataset_collate_fn)

model_config["TOKENS"] = 16
model_config["SRC_LEN"] = dataset.SRC_LEN
model_config["TGT_LEN"] = dataset.TGT_LEN

model = frt.FRT(model_config)
model.device = device
model.to(device)

model.load_state_dict(torch.load(args.params, map_location=torch.device(device)))
model.eval()

print(dataset.dictionary.idx2word)
print(dataset.dictionary.word2idx)

correct = 0
total = 0

with tqdm(total=len(dataset)) as prog:
	for (src_indicies, src_padding_mask, tgt_indicies, tgt_padding_mask) in dataloader:
		src_indicies = src_indicies.to(device)
		tgt_indicies = tgt_indicies.to(device)
		src_padding_mask = src_padding_mask.to(device)

		question_strings = [ q_str.split(dataset.PADDING)[0] for q_str in dataset.tensor2text(src_indicies)]
		target_strings = [ tgt_str.split(dataset.END)[0] for tgt_str in dataset.tensor2text(tgt_indicies)]

		print(question_strings)
		print(target_strings)

		print("Target Indicies: ")
		print(tgt_indicies)
		output = model.predict(src_indicies, src_padding_mask, dataset.dictionary.word2idx[dataset.START])

		output_strings = [ out_str.split(dataset.END)[0] for out_str in dataset.tensor2text(output)]

		for j in range(len(target_strings)):
			question = question_strings[j]
			pred = output_strings[j]
			actual = target_strings[j]

			#print("Q: {} , A: {}".format(question, actual), file=f)
			#print("Got: '{}' {}\n".format(pred, "correct" if actual == pred else "wrong"), file=f)
			correct += (actual == pred)
			total += 1
		prog.update(dataloader.batch_size)
#print("{} Correct out of {} total. {:.3f}% accuracy".format(correct, total, correct/total * 100), file=f)
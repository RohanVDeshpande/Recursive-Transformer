import math
import random
from termcolor import colored
import json
import utils
from utils import AverageMeter
import os
import time

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
parser.add_argument('--mode', type=str, default=None, required=True,
                    help="Choose train mode vs. test mode",
					choices=["train", "test"])
parser.add_argument('--data', type=str, required=True,
                    help='Dataset path')
parser.add_argument('--model-config', type=str, required=True,
                    help='Model json config')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')

args = parser.parse_args()

assert os.path.isdir("output/"), "You need a folder called 'output' in order to save model data / test predictions"
assert os.path.isdir("tb/"), "You need a folder called 'tb' for tensorboard"

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
  model_path = "output/{}.pt".format(model_config["NAME"])
  prediction_path = "output/{}_{}.txt".format(model_config["NAME"], os.path.splitext(os.path.basename(args.data))[0])

  if args.mode == "train" and os.path.exists(model_path):
    utils.confirm("Re-training the model will overwrite the following file: {}".format(model_path))
  elif args.mode == "test" and os.path.exists(prediction_path):
   utils.confirm("Re-testing will cause the prediction output file to be overwritten: {}".format(prediction_path))

  with open(model_config["DATASET_CONFIG"]) as f:
  	dataset_config = json.load(f)

dataset = data.Dataset(dataset_config)
dataset.buildDataset(args.data)
dataset.device = device

dataloader = DataLoader(dataset, batch_size=dataset_config["BATCH_SIZE"], shuffle=dataset_config["SHUFFLE"], num_workers=dataset_config["WORKERS"],
           pin_memory=dataset_config["PIN_MEMORY"], prefetch_factor=dataset_config["PREFETCH_FACTOR"],
           persistent_workers=True, collate_fn=data.dataset_collate_fn)

model_config["TOKENS"] = dataset.tokens()
model_config["TGT_LEN"] = dataset.TGT_LEN

model = frt.FRT(model_config)
model.device = device
model.to(device)

if args.mode == "train":
	assert model_path is not None, "Model path not set up"

	writer = SummaryWriter("tb/", comment=utils.config2comment(model_config, dataset_config))

	if model_config["OPTIMIZER"] == "ADAM":
		optimizer = optim.Adam(model.parameters(), lr=model_config["LR"])
	criterion = nn.NLLLoss()

	EPOCHS = model_config["EPOCHS"]

	model.train()
	iteration=0
	for epoch in range(EPOCHS):
		epoch_loss = 0.
		with tqdm(total=len(dataset)) as prog:
			batch_time = AverageMeter()
			data_time = AverageMeter()
			update_time = AverageMeter()

			batch_start_time = time.time()
			for i, (src_indicies, src_padding_mask, tgt_indicies, tgt_padding_mask) in enumerate(dataloader):

				src_indicies = src_indicies.to(device)
				#src_padding_mask = src_padding_mask.to(device)
				tgt_indicies = tgt_indicies.to(device)
				#tgt_padding_mask = tgt_padding_mask.to(device)
				#src_indicies = src_indicies.transpose(0, 1)
				#tgt_indicies = tgt_indicies.transpose(0, 1)

				data_time.update(time.time() - batch_start_time) # data loading time

				update_start_time = time.time()

				optimizer.zero_grad()
				output, tgt = model(src_indicies, tgt_indicies)
				loss = criterion(output, tgt.view(-1))
				loss.backward()
				optimizer.step()

				update_time.update(time.time() - update_start_time)

				epoch_loss += loss.item()
				prog.update(dataloader.batch_size)
				
				batch_time.update(time.time() - batch_start_time)
				batch_start_time = time.time()

				if i % 16 == 0:
					print('avg data load time: {}\n'
						  'avg update time: {}\n'
						  'avg total batch time: {}'.format(data_time.avg, update_time.avg, batch_time.avg))
				#writer.add_scalar("Loss/train", loss.item(), iteration)
				#iteration += 1
			epoch_loss /= len(dataset)
			print("Epoch {}, Loss: {}".format(epoch, epoch_loss))

	torch.save(model.state_dict(), model_path)

elif args.mode == "test":
	assert model_path is not None, "Model path not set up"
	model.load_state_dict(torch.load(model_path))
	model.eval()

	correct = 0
	total = 0

	with open(prediction_path, "w") as f:
		with tqdm(total=len(dataset)) as prog:
			for (src_indicies, src_padding_mask, tgt_indicies, tgt_padding_mask) in dataloader:
				output = model.predict(src_indicies, src_padding_mask, dataset.dictionary.word2idx[dataset.START])

				question_strings = [ q_str.split(dataset.PADDING)[0] for q_str in dataset.tensor2text(src_indicies)]
				target_strings = [ tgt_str.split(dataset.END)[0] for tgt_str in dataset.tensor2text(tgt_indicies)]
				output_strings = [ out_str.split(dataset.END)[0] for out_str in dataset.tensor2text(output)]

				for j in range(len(target_strings)):
					question = question_strings[j]
					pred = output_strings[j]
					actual = target_strings[j]

					print("Q: {} , A: {}".format(question, actual), file=f)
					print("Got: '{}' {}\n".format(pred, "correct" if actual == pred else "wrong"), file=f)
					correct += (actual == pred)
					total += 1
				prog.update(1)
		print("{} Correct out of {} total. {:.3f}% accuracy".format(correct, total, correct/total * 100), file=f)

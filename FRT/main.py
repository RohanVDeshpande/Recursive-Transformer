import math
import random
from termcolor import colored
import json
from . import utils
from .utils import AverageMeter
import os
import sys
import time
import datetime
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
from . import frt
from . import runner
from . import data

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(description='Forced Recursive Transformer')
parser.add_argument('--mode', type=str, default=None, required=True,
                    help="Choose train mode vs. test mode vs. finetune",
					choices=["train", "test", "finetune"])
parser.add_argument('--data', type=str, required=True,
                    help='Dataset path (for training or testing)')
parser.add_argument('--validation', type=str, help='Validation set path')
parser.add_argument('--model-config', type=str, required=True,
                    help='Model json config')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--params', type=str, help='Model params path for finetuning or testing')
parser.add_argument('--dict', type=str, help='Dictionary path for finetuning or testing')
parser.add_argument('--dry-run', action='store_true', help='enable dry run, one or two batches')
parser.add_argument('--logdir', type=str, help='TensorBoard logging directory', default='tb')
parser.add_argument('--test-type', type=str, choices=["step", "recursive"])

args = parser.parse_args()

assert os.path.isdir("output/"), "You need a folder called 'output' in order to save test predictions"
assert os.path.isdir(args.logdir), "Your provided logdir '{}' is not a directory".format(logdir)
assert os.path.isdir("output/dict/"), "You need a folder called 'dict' in order to save/load a dictionary"
assert os.path.isdir("checkpoints/"), "You need a folder called 'checkpoints' in order to save model params"

if args.mode == 'train':
	assert args.validation, "You need to provide a validation set (--validation) for training"
if args.mode == 'finetune':
	assert args.dict, "You need to provide a dictionary path (--dict) for finetuning"
	assert args.params, "You need to provide a params path (--params) for finetuning"
if args.mode == 'testing':
	assert args.params, "You need to provide a params path (--params) for testing"
	assert args.test_type, "You need to provide test type (--test-type) for testing"

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
  checkpoint_dir = "checkpoints/{}".format(model_config["NAME"])
  dictionary_in_path = args.dict if args.dict else "output/dict/{}_dict.json".format(model_config["NAME"])
  dictionary_out_path = "output/dict/{}_dict.json".format(model_config["NAME"])
  prediction_path = "output/{}_{}.txt".format(model_config["NAME"], os.path.splitext(os.path.basename(args.data))[0])
  tb_log_path = os.path.join(args.logdir, model_config["NAME"])
  print('logging to', tb_log_path)

  if args.mode == "train" and os.path.exists(checkpoint_dir):
    utils.confirm("Re-training the model will overwrite the checkpoints in {}".format(checkpoint_dir))
  elif args.mode == "test" and os.path.exists(prediction_path):
   utils.confirm("Re-testing will cause the prediction output file to be overwritten: {}".format(prediction_path))

  with open(model_config["DATASET_CONFIG"]) as f:
  	dataset_config = json.load(f)

dataset = data.Dataset(dataset_config)
if args.mode == "test" or args.mode == "finetune":
	print("Loading dictionary from: {}".format(dictionary_in_path))
	dataset.loadDictionary(dictionary_in_path)
	if args.mode == "finetune":
		print('unfreezing dataset for finetune')
		dataset.dictionary.freeze_dict = False
	else:
		dataset.RANDOMIZE_LEFT_PADDING = False
dataset.buildDataset(args.data)
dataset.device = device

dataloader = DataLoader(dataset, batch_size=dataset_config["BATCH_SIZE"], shuffle=dataset_config["SHUFFLE"], num_workers=dataset_config["WORKERS"],
           pin_memory=dataset_config["PIN_MEMORY"], prefetch_factor=dataset_config["PREFETCH_FACTOR"],
           persistent_workers=True, collate_fn=data.dataset_collate_fn)

model_config["TOKENS"] = dataset.tokens()
model_config["SRC_LEN"] = dataset.SRC_LEN
model_config["TGT_LEN"] = dataset.TGT_LEN

model = frt.FRT(model_config)
model.device = device
model.to(device)

if args.mode == "train" or args.mode == "finetune":
	assert checkpoint_dir is not None, "Model path not set up"
	if not os.path.exists(checkpoint_dir):
		os.mkdir(checkpoint_dir)

	if args.mode == "finetune":
		model.load_state_dict(torch.load(args.params, map_location=torch.device(device)))
	val_dataset = data.Dataset(dataset)				# configure validation dataset object from trainging dataset object's config
													# this allows dictionary to be shared
	if args.mode == "finetune":
		print('unfreezing validation set for finetune')
		val_dataset.dictionary.freeze_dict = False
	dataset.saveDictionary(dictionary_out_path)
	val_dataset.buildDataset(args.validation)
	val_dataset.device = device
	val_dataloader = DataLoader(val_dataset, batch_size=dataset_config["BATCH_SIZE"], shuffle=dataset_config["SHUFFLE"], num_workers=dataset_config["WORKERS"],
           pin_memory=dataset_config["PIN_MEMORY"], prefetch_factor=dataset_config["PREFETCH_FACTOR"],
           persistent_workers=True, collate_fn=data.dataset_collate_fn)

	tb_writer = SummaryWriter(tb_log_path, comment=utils.config2comment(model_config, dataset_config))

	if model_config["OPTIMIZER"] == "ADAM":
		optimizer = optim.Adam(model.parameters(), lr=model_config["LR"])
	elif model_config["OPTIMIZER"] == "ADAMW":
		optimizer = optim.AdamW(model.parameters(), lr=model_config["LR"])
	
	weights = torch.ones(model.TOKENS)
	weights[dataset.dictionary.word2idx[dataset.START1]] = 0
	weights[dataset.dictionary.word2idx[dataset.START2]] = 0
	weights[dataset.dictionary.word2idx[dataset.PADDING]] = 0
	weights = weights.to(device)

	criterion = nn.NLLLoss(weight=weights)
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=model_config["LR_SCHEDULER_DECAY"], verbose=True)

	EPOCHS = 1 if args.dry_run else model_config["EPOCHS"]

	model.train()
	iteration=0
	try:
		for epoch in range(EPOCHS):
			with tqdm(total=len(dataset)) as prog:
				batch_time = AverageMeter()
				data_time = AverageMeter()
				update_time = AverageMeter()

				batch_start_time = time.time()
				for i, (src_indicies, src_padding_mask, tgt_indicies, tgt_padding_mask) in enumerate(dataloader):
					if (args.dry_run and i == 5):
						# 'dry run' only runs 1 epoch with 5 bathes
						break

					src_indicies = src_indicies.to(device)
					src_padding_mask = src_padding_mask.to(device)
					tgt_indicies = tgt_indicies.to(device)
					tgt_padding_mask = tgt_padding_mask.to(device)

					data_time.update(time.time() - batch_start_time) # data loading time

					update_start_time = time.time()

					optimizer.zero_grad()
					output, tgt = model(src_indicies, tgt_indicies, src_padding_mask, tgt_padding_mask)
					loss = criterion(output, tgt.view(-1))
					loss.backward()
					optimizer.step()

					update_time.update(time.time() - update_start_time)
					prog.update(dataloader.batch_size)
					
					batch_time.update(time.time() - batch_start_time)
					batch_start_time = time.time()

					tb_writer.add_scalar("Loss/train", loss.item(), iteration)
					iteration += dataset.BATCH_SIZE

					# calculate and log validation loss every 1/5 of a dataset pass
					if iteration % (len(dataset)//5) == 0:
						model.eval()
						print('VALIDATING ITER {}'.format(iteration))
						with torch.no_grad():
							epoch_val_loss = 0
							for (src_indicies, src_padding_mask, tgt_indicies, tgt_padding_mask) in val_dataloader:
								src_indicies = src_indicies.to(device)
								src_padding_mask = src_padding_mask.to(device)
								tgt_indicies = tgt_indicies.to(device)
								tgt_padding_mask = tgt_padding_mask.to(device)

								output, tgt = model(src_indicies, tgt_indicies, src_padding_mask, tgt_padding_mask)
								loss = criterion(output, tgt.view(-1))
								epoch_val_loss += loss.item()
							
							epoch_val_loss /= len(val_dataloader)
							tb_writer.add_scalar("Loss/validation", epoch_val_loss, iteration)
						model.train()
						scheduler.step()

				# at end of epoch, save checkpoint
				checkpoint_path = os.path.join(checkpoint_dir, '{}_epoch{}.pt'.format(model_config["NAME"], epoch))
				torch.save(model.state_dict(), checkpoint_path)
	except:
		print('-' * 89)
		print('Exiting from training early')
		checkpoint_path = os.path.join(checkpoint_dir, '{}_epoch{}_terminated.pt'.format(model_config["NAME"], epoch))
		torch.save(model.state_dict(), checkpoint_path)
		traceback.print_exc()
		print("Training killed")
	

elif args.mode == "test":
	assert args.params is not None, "Model params path not set up"
	model.load_state_dict(torch.load(args.params, map_location=torch.device(device)))
	model.eval()

	if args.test_type == "step":
		runner.test(model, dataloader, dataset, prediction_path)
	elif args.test_type == "recursive":
		runner.testRecursive(model, dataset, prediction_path)

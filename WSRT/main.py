import math
import random
import json
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
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from . import data
from . import wsrt
from .utils import AverageMeter
from . import utils


parser = argparse.ArgumentParser(description='Weakly Supervised Recursive Transformer')
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

args = parser.parse_args()

assert os.path.isdir("output/"), "You need a folder called 'output' in order to save test predictions"
assert os.path.isdir("tb/"), "You need a folder called 'tb' for tensorboard"
assert os.path.isdir("output/dict/"), "You need a folder called 'dict' in order to save/load a dictionary"
assert os.path.isdir("checkpoints/"), "You need a folder called 'checkpoints' in order to save model params"

if args.mode == 'train':
	assert args.validation, "You need to provide a validation set (--validation) for training"
if args.mode == 'finetune':
	assert args.dict, "You need to provide a dictionary path (--dict) for finetuning"
	assert args.params, "You need to provide a params path (--params) for finetuning"
if args.mode == 'testing':
	assert args.params, "You need to provide a params path (--params) for testing"

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
	dictionary_path = args.dict if args.dict else "output/dict/{}_dict.json".format(model_config["NAME"])
	prediction_path = "output/{}_{}.txt".format(model_config["NAME"], os.path.splitext(os.path.basename(args.data))[0])
	tb_log_path = "tb/{}".format(model_config["NAME"])

	if args.mode == "train" and os.path.exists(checkpoint_dir):
		utils.confirm("Re-training the model will overwrite the checkpoints in {}".format(checkpoint_dir))
	elif args.mode == "test" and os.path.exists(prediction_path):
	 utils.confirm("Re-testing will cause the prediction output file to be overwritten: {}".format(prediction_path))

	with open(model_config["DATASET_CONFIG"]) as f:
		dataset_config = json.load(f)

dataset = data.Dataset(dataset_config)
if args.mode == "test" or args.mode == "finetune":
	print("Loading dictionary from: {}".format(dictionary_path))
	dataset.loadDictionary(dictionary_path)
	if args.mode == "finetune":
			dataset.dictionary.freeze_dict = False
dataset.buildDataset(args.data)
dataset.device = device

dataloader = DataLoader(dataset, batch_size=dataset_config["BATCH_SIZE"], shuffle=dataset_config["SHUFFLE"], num_workers=dataset_config["WORKERS"],
					 pin_memory=dataset_config["PIN_MEMORY"], prefetch_factor=dataset_config["PREFETCH_FACTOR"],
					 persistent_workers=True, collate_fn=data.dataset_collate_fn)

model_config["TOKENS"] = dataset.tokens()
model_config["SRC_LEN"] = dataset.SRC_LEN
model_config["TGT_LEN"] = dataset.TGT_LEN

model = wsrt.WSRT(model_config)
model.device = device
model.to(device)

if args.mode == "train" or args.mode == "finetune":
	assert checkpoint_dir is not None, "Model path not set up"
	if not os.path.exists(checkpoint_dir):
		os.mkdir(checkpoint_dir)

	if args.mode == "finetune":
		model.load_state_dict(torch.load(args.params), strict=False)
	val_dataset = data.Dataset(dataset)				# configure validation dataset object from trainging dataset object's config
													# this allows dictionary to be shared
	if args.mode == "finetune":
		val_dataset.dictionary.freeze_dict = False
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
	token_weights = torch.ones(model_config["TOKENS"], device=device)
	token_weights[dataset.dictionary.word2idx[dataset.PADDING]] = 0.02
	criterion = nn.NLLLoss(weight=token_weights)
	scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=model_config["LR_SCHEDULER_DECAY"], verbose=True)

	EPOCHS = 3 if args.dry_run else model_config["EPOCHS"]

	model.train()
	iteration=0
	try:
		for epoch in range(EPOCHS):
			with tqdm(total=len(dataset)) as prog:
				prog.set_description("Training")
				batch_time = AverageMeter()
				data_time = AverageMeter()
				update_time = AverageMeter()

				batch_start_time = time.time()
				for i, (src_indicies, tgt_indicies, WSRT_steps) in enumerate(dataloader):
					if model.RANDOMIZE_STEPS:
						WSRT_steps = random.randint(WSRT_steps, int(WSRT_steps * model.RANDOMIZE_STEPS_SCALE_FACTOR))
					# print(WSRT_steps)
					# print(src_indicies)
					# print(src_indicies.shape)
					# print(tgt_indicies)
					# print(tgt_indicies.shape)
					if (args.dry_run and i == 5):
						# 'dry run' only runs 1 epoch with 5 batches
						break

					src_indicies = src_indicies.to(device)
					tgt_indicies = tgt_indicies.to(device)

					data_time.update(time.time() - batch_start_time) # data loading time

					update_start_time = time.time()

					optimizer.zero_grad()
					output, tgt = model(src_indicies, tgt_indicies, WSRT_steps, dataset.dictionary.word2idx[dataset.START], dataset.dictionary.word2idx[dataset.END], tgt_indicies.shape[0])
					# print(output)
					# print(output.shape)
					# print(tgt_indicies.view(-1).shape)
					loss = criterion(output, tgt.view(-1))
					loss.backward()
					optimizer.step()

					update_time.update(time.time() - update_start_time)
					prog.update(dataloader.batch_size)
					
					batch_time.update(time.time() - batch_start_time)
					batch_start_time = time.time()

					tb_writer.add_scalar("Loss/train", loss.item(), iteration)
					iteration += 1

					# calculate and log validation loss every 1/5 of a dataset pass
					if (args.dry_run and i == 4) or iteration % (len(dataloader)//5) == 0:
						model.eval()
						with torch.no_grad():
							epoch_val_loss = 0
							with tqdm(total=len(val_dataset)) as val_prog:
								val_prog.set_description("Validating")
								for j, (src_indicies, tgt_indicies, WSRT_steps) in enumerate(val_dataloader):
									if (args.dry_run and j == 1):
										# 'dry run' only runs 1 epoch with 5 bathes
										break
									if model.RANDOMIZE_STEPS:
										WSRT_steps = random.randint(WSRT_steps, int(WSRT_steps * model.RANDOMIZE_STEPS_SCALE_FACTOR))
									src_indicies = src_indicies.to(device)
									tgt_indicies = tgt_indicies.to(device)
									
									output, tgt = model(src_indicies, tgt_indicies, WSRT_steps, dataset.dictionary.word2idx[dataset.START], dataset.dictionary.word2idx[dataset.END], tgt_indicies.shape[0])
									loss = criterion(output, tgt.view(-1))
									epoch_val_loss += loss.item()
									val_prog.update(val_dataloader.batch_size)
								
								epoch_val_loss /= len(val_dataloader)
								tb_writer.add_scalar("Loss/validation", epoch_val_loss, iteration)
								# at end of epoch, save checkpoint
								checkpoint_path = os.path.join(checkpoint_dir, '{}_epoch{}_iter{}.pt'.format(model_config["NAME"], epoch, iteration))
								torch.save(model.state_dict(), checkpoint_path)
						model.train()
						scheduler.step()

				# at end of epoch, save checkpoint
				checkpoint_path = os.path.join(checkpoint_dir, '{}_epoch{}_end.pt'.format(model_config["NAME"], epoch))
				torch.save(model.state_dict(), checkpoint_path)
	except:
		traceback.print_exc() 
		print('\n', '-' * 89)
		print('Exiting from training early')
		checkpoint_path = os.path.join(checkpoint_dir, '{}_epoch{}_terminated.pt'.format(model_config["NAME"], epoch))
		torch.save(model.state_dict(), checkpoint_path)
	# print('Saving model to {}'.format(model_path))
	# torch.save(model.state_dict(), model_path)
	dataset.saveDictionary(dictionary_path)

elif args.mode == "test":
	assert args.params is not None, "Model params path not set up"
	model.load_state_dict(torch.load(args.params, map_location=torch.device(device)))
	model.eval()
	with torch.no_grad():
		correct = 0
		total = 0

		with open(prediction_path, "w") as f:
			with tqdm(total=len(dataset)) as prog:
				for (src_indicies, tgt_indicies, WSRT_steps) in dataloader:
					if model.RANDOMIZE_STEPS:
						WSRT_steps = random.randint(WSRT_steps, int(WSRT_steps * model.RANDOMIZE_STEPS_SCALE_FACTOR))
					src_indicies = src_indicies.to(device)
					tgt_indicies = tgt_indicies.to(device)

					output = model.predictFinal(src_indicies, dataset.dictionary.word2idx[dataset.START], WSRT_steps)

					question_strings = dataset.tensor2text(src_indicies)
					target_strings = dataset.tensor2text(tgt_indicies)
					output_strings = dataset.tensor2text(output)

					for j in range(len(target_strings)):
						question = question_strings[j]
						pred = output_strings[j]
						actual = target_strings[j]

						print("Q: {} , A: {}".format(question, actual), file=f)
						print("Got: '{}' steps: {} {}\n".format(pred, WSRT_steps, "correct" if actual == pred else "wrong"), file=f)
						correct += (actual == pred)
						total += 1
					prog.update(dataloader.batch_size)
			print("{} Correct out of {} total. {:.3f}% accuracy".format(correct, total, correct/total * 100), file=f)

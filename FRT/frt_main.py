import math
import random
from termcolor import colored
import json
import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import frt

import data
from tqdm import tqdm
#from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
parser.add_argument('--data', type=str,
                    help='Dataset path')
parser.add_argument('--model-config', type=str,
                    help='Model json config')
parser.add_argument('--train', action='store_true',
                    help='Model json config')
parser.add_argument('--test', action='store_true',
                    help='Model json config')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')

args = parser.parse_args()

if (not args.train and not args.test) or (args.train and args.test):
	print("Must select either --train or --test")
	sys.exit(1)

# Set the random seed manually for reproducibility.
# torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
    	utils.confirm("You have a CUDA device, so you should probably run with --cuda.".format(args.save))
    else:
    	print("Using CUDA device")

device = torch.device("cuda" if args.cuda else "cpu")

if os.path.exists(args.save):
    utils.confirm("The following file will be overwritten when the model saves: {}".format(args.save))


with open(args.model_config) as f:
  model_config = json.load(f)

  with open(model_config["DATASET_CONFIG"]) as f:
  	dataset_config = json.load(f)

dataset = data.Dataset(dataset_config)
dataset.buildDataset(args.data)
dataset.device = device

model_config["TOKENS"] = dataset.tokens()
model_config["TGT_LEN"] = dataset.TGT_LEN

model = frt.FRT(model_config)
model.to(device)

if args.train:
	if model_config["OPTIMIZER"] == "ADAM":
		optimizer = optim.Adam(model.parameters(), lr=model_config["LR"])
	criterion = nn.NLLLoss(reduction='sum')

	EPOCHS = model_config["EPOCHS"]
	#tb_writer = SummaryWriter("save/" + model_config["NAME"] + "/")

	model.train()
	for epoch in range(EPOCHS):
		epoch_loss = 0.
		with tqdm(total=train_dataset.batches()//2) as prog:
			for batch in range(train_dataset.batches()//2):
				# print("Epoch {} Batch {} / {}".format(epoch, batch, train_dataset.batches()))
				optimizer.zero_grad()
				output, tgt = model(*train_dataset.get_data(batch))
				loss = criterion(output, tgt.view(-1))
				loss.backward()
				optimizer.step()
				epoch_loss += loss.item()
				prog.update(1)
			print("Epoch {}, Loss: {}".format(epoch, epoch_loss))

elif args.test:
	model.eval()

	correct = 0
	total = 0

	with tqdm(total=dataset.batches()) as prog:
		for i in range(dataset.batches()):

			src_indicies, src_padding_mask, tgt_indicies, tgt_padding_mask = dataset.get_data(i)
			output = model.predict(src_indicies, src_padding_mask, test_dataset.dictionary.word2idx[dataset.START])

			question_strings = [ q_str.split(dataset.PADDING)[0] for q_str in dataset.tensor2text(src_indicies)]
			target_strings = [ tgt_str.split(dataset.END)[0] for tgt_str in dataset.tensor2text(tgt_indicies)]
			output_strings = [ out_str.split(dataset.END)[0] for out_str in dataset.tensor2text(output)]


			for j in range(len(target_strings)):
				question = question_strings[j]
				pred = output_strings[j]
				actual = target_strings[j]

				print("Q: {} , A: {}".format(question, actual))
				print(colored("Got: '{}'".format(pred), "blue" if actual == pred else "red"))
				print()
				correct += (actual == pred)
				total += 1
			prog.update(1)
	print("{} Correct out of {} total. {:.3f}% accuracy".format(correct, total, correct/total * 100))
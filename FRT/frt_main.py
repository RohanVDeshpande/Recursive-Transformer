import math
import random
import pdb		# for debugging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import frt

import data

from termcolor import colored
from tqdm import tqdm

# EPOCHS = 200
# TRAIN_PATH = "mathematics_dataset/raw_data_tsv/super_simple_add_subtract.tsv"
# TEST_PATH = "mathematics_dataset/raw_data_tsv/super_simple_add_subtract.tsv"
EPOCHS = 5	# delete this after debugging
TRAIN_PATH = "mathematics_dataset/ssas.tsv"
TEST_PATH = "mathematics_dataset/ssas.tsv"

dataset_config = {
	"START" : u"\u2361",				# ह
	"SRC_TGT_SEP": u"\u2358",			# श
	"TGT_LOOP_SEP" : u"\u2325",			# क
	"END" : u"\u2352",					# र
	"PADDING" : u"\u2340",				# त
	"SRC_LEN" : 32,
	"TGT_LEN" : 32,
	"BATCH_SIZE": 16
}

train_dataset = data.Dataset(dataset_config)
train_dataset.buildDataset(TRAIN_PATH)

frt_config = {
	"TOKENS" : train_dataset.tokens(),
	"TGT_LEN": train_dataset.TGT_LEN,
	"FEATURES" : 32,
	"HEADS" : 8,
	"ENC_LAYERS" : 4,
	"DEC_LAYERS" : 4,
	"FEED_FORWARD" : 64
}


model = frt.FRT(frt_config)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.NLLLoss()

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

model.eval()
print()

test_dataset = data.Dataset(train_dataset)
test_dataset.buildDataset(TEST_PATH)

correct = 0
total = 0

with tqdm(total=test_dataset.batches()) as prog:
	for i in range(test_dataset.batches()):

		src_indicies, src_padding_mask, tgt_indicies, tgt_padding_mask = test_dataset.get_data(i)
		output = model.predict(src_indicies, src_padding_mask, test_dataset.dictionary.word2idx[test_dataset.START])
		# breakpoint()
		question_strings = [ q_str.split(test_dataset.PADDING)[0] for q_str in test_dataset.tensor2text(src_indicies)]
		target_strings = [ tgt_str.split(test_dataset.END)[0] for tgt_str in test_dataset.tensor2text(tgt_indicies)]
		output_strings = [ out_str.split(test_dataset.END)[0] for out_str in test_dataset.tensor2text(output)]


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
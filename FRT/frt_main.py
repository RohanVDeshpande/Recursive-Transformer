import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import frt

import data

from termcolor import colored

EPOCHS = 5
TRAIN_PATH = "mathematics_dataset/raw_data_tsv/train-easy/arithmetic__add_or_sub.tsv"
TEST_PATH = "mathematics_dataset/raw_data_tsv/train-easy/arithmetic__add_or_sub.tsv"

dataset_config = {
	"SRC_TGT_SEP": u"\u2358",			# श
	"TGT_LOOP_SEP" : u"\u2325",			# क
	"END" : u"\u2352",					# र
	"PADDING" : u"\u2340",				# त
	"SRC_LEN" : 16,
	"TGT_LEN" : 16,
	"BATCH_SIZE": 16
}

train_dataset = data.Dataset(dataset_config)
train_dataset.buildDataset(TRAIN_PATH)

frt_config = {
	"TOKENS" : train_dataset.tokens(),
	"TGT_LEN": train_dataset.TGT_LEN,
	"FEATURES" : 16,
	"HEADS" : 2,
	"ENC_LAYERS" : 2,
	"DEC_LAYERS" : 2,
	"FEED_FORWARD" : 64
}


model = frt.FRT(frt_config)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.NLLLoss()

model.train()
for epoch in range(EPOCHS):
	epoch_loss = 0.
	for batch in range(train_dataset.batches()//2):
		#print("Epoch {} Batch {} / {}".format(epoch, batch, train_dataset.batches()))
		optimizer.zero_grad()
		output, tgt = model(*train_dataset.get_data(batch))
		loss = criterion(output, tgt.view(-1))
		loss.backward()
		optimizer.step()
		epoch_loss += loss.item()
	print("Epcoh {}, Loss: {}".format(epoch, epoch_loss))


model.eval()
print()

test_dataset = data.Dataset(train_dataset)
test_dataset.buildDataset(TEST_PATH)

correct = 0
total = 0

for i in range(test_dataset.batches()//2, test_dataset.batches()):

	src_indicies, src_padding_mask, tgt_indicies, _ = test_dataset.get_data(i)
	output = model.predict(src_indicies, src_padding_mask)

	question_strings = [ q_str.split(dataset_config.PADDING)[0] for q_str in test_dataset.tensor2text(src_indicies)]
	target_strings = [ tgt_str.split(dataset_config.END)[0] for tgt_str in test_dataset.tensor2text(tgt_indicies)]
	output_strings = [ out_str.split(dataset_config.END)[0] for out_str in test_dataset.tensor2text(output)]


	for j in range(target_strings):
		question = question_strings[j]
		pred = output_strings[j]
		actual = target_strings[j]

		print("Q: {} , A: {}".format(question, actual))
		print(colored("Got: '{}'".format(pred), "blue" if actual == pred else "red"))
		print()
		correct += (actual == pred)
		total += 1
print("{} Correct out of {} total. {:.3f}% accuracy".format(correct, total, correct/total * 100))
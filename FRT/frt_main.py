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
test_dataset.BATCH_SIZE = 1
test_dataset.buildDataset(TEST_PATH)

correct = 0
total = 0

printed = 0
for i in range(test_dataset.batches()//2, test_dataset.batches()):

	output, tgt = model(*test_dataset.get_data(i))
	got=torch.argmax(output, 1)
	#print(got)
	#print(tgt_ref)

	output_str = test_dataset.tensor2text(tgt.view(-1))
	got_str = test_dataset.tensor2text(got.view(-1))

	if printed < 100:
		print("Q: {} , A: {}".format(test_dataset.questions[i].split(test_dataset.PADDING)[0],
													  test_dataset.answers[i].split(test_dataset.TGT_LOOP_SEP)[0]))
		print("Expected: '{}'".format(output_str))
		print(colored("Got: '{}'".format(got_str), "blue" if output_str == got_str else "red"))
		print()
		printed += 1
	correct += (output_str == got_str)
	total += 1
print("{} Correct out of {} total. {:.3f}% accuracy".format(correct, total, correct/total * 100))
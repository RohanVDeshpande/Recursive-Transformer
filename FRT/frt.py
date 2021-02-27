import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data
import random
from termcolor import colored

SRC_TGT_SEP = u"\u2358"			# श
TGT_LOOP_SEP = u"\u2325"		# क
END = u"\u2352"					# र
PADDING = u"\u2340"				# त
LEN = 16
BLOCKSIZE = 4
FEATURES = 32


dataset = []

def make_q_a(num1, num2):
	q = "{}+{}".format(num1, num2)
	q += (LEN - len(q)) * PADDING
	a = "{}{}1{}".format(num1 + num2, TGT_LOOP_SEP, END)
	a += (LEN - len(a)) * PADDING
	return (q, a)


for i in range(5 * BLOCKSIZE):
	num1 = random.randint(0, 20)
	num2 = random.randint(0, 20)
	dataset.append(make_q_a(num1, num2))

d = data.Dictionary()

for sample in dataset:
	for c in sample[0]:
		d.add_word(c)
	for c in sample[1]:
		d.add_word(c)


def text2tensor(dictionary, text):
	return torch.tensor([ dictionary.word2idx[c] for c in text], dtype=torch.long)

def tensor2text(dictionary, t):
	if len(t.shape) == 1:
		return "".join([dictionary.idx2word[idx] for idx in t])
	else:
		return ""



class FRT(nn.Module):
	""" Forced Recursive Transformer """

	def __init__(self, config):
		super().__init__()

		#src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask
		self.embed = nn.Embedding(config["TOKENS"], config["FEATURES"])
		self.transformer = nn.Transformer(d_model=config["FEATURES"], nhead=config["HEADS"], num_encoder_layers=config["ENC_LAYERS"], num_decoder_layers=config["DEC_LAYERS"], dim_feedforward=config["FEED_FORWARD"])
		self.log_softmax = nn.LogSoftmax()
		self.apply(self._init_weights)
		print("number of parameters: {}".format(sum(p.numel() for p in self.parameters())))

	def _init_weights(self, module):
		if isinstance(module, nn.Embedding):
			module.weight.data.normal_(mean=0.0, std=0.02)

	def forward(self, srctext, tgttext):
		src_indicies = text2tensor(d, srctext)
		src = torch.unsqueeze(self.embed(src_indicies), 1)
		src_padding_mask = torch.tensor([ float('-inf') if c == PADDING else 0 for c in  srctext]).view(1, -1)

		tgt_indicies = text2tensor(d, tgttext)
		tgt = torch.unsqueeze(self.embed(tgt_indicies), 1)
		tgt_padding_mask = torch.tensor([ float('-inf') if c == PADDING else 0 for c in  tgttext]).view(1, -1)
		#print(src.shape, tgt.shape)
		output = self.transformer(src, tgt,
								  tgt_mask=self.transformer.generate_square_subsequent_mask(LEN),
								  src_key_padding_mask=src_padding_mask,
								  tgt_key_padding_mask=tgt_padding_mask).view(-1, FEATURES)
		output = self.log_softmax(output)
		#print(output)

		return output, tgt_indicies



# d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6, num_decoder_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = 'relu'
config = {}
config["FEATURES"] = FEATURES
config["TOKENS"] = d.len()
config["HEADS"] = 2
config["ENC_LAYERS"] = 2
config["DEC_LAYERS"] = 2
config["FEED_FORWARD"] = 64

model = FRT(config)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.NLLLoss()

model.train()
for epoch in range(40):
	epoch_loss = 0.
	for sample in dataset:
	    optimizer.zero_grad()
	    output, tgt_ref = model(sample[0], sample[1])
	    #print(output.shape)
	    #print(tgt.shape)
	    loss = criterion(output, tgt_ref)
	    loss.backward()
	    optimizer.step()
	    epoch_loss += loss.item()
	print("Epcoh {}, Loss: {}".format(epoch, epoch_loss))

model.eval()
print()

training_questions = {sample[0] for sample in dataset}

for i in range(15):
	num1 = random.randint(0, 50)
	num2 = random.randint(0, 50)
	q, a = make_q_a(num1, num2)
	output, tgt_ref = model(q, a)
	got=torch.argmax(output, 1)
	#print(got)
	#print(tgt_ref)
	output_str = tensor2text(d, tgt_ref)
	got_str = tensor2text(d, got)

	print("Q: {}+{} , A: {}. Seen before: {}".format(num1, num2, num1 + num2, q in training_questions))
	print("Expected: '{}'".format(output_str))
	print(colored("Got: '{}'".format(got_str), "blue" if output_str == got_str else "red"))
	print()
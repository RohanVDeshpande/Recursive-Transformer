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
BATCHES = 10

dataset_questions = []
dataset_answers = []

def make_q_a(num1, num2):
	q = "{}+{}".format(num1, num2)
	q += (LEN - len(q)) * PADDING
	a = "{}{}1{}".format(num1 + num2, TGT_LOOP_SEP, END)
	a += (LEN - len(a)) * PADDING
	return (q, a)


for i in range(BATCHES * BLOCKSIZE):
	num1 = random.randint(0, 40)
	num2 = random.randint(0, 40)
	q, a = make_q_a(num1, num2)
	dataset_questions.append(q)
	dataset_answers.append(a)

d = data.Dictionary()

for q in dataset_questions:
	for c in q:
		d.add_word(c)
for a in dataset_answers:
	for c in a:
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
		self.lin_out = nn.Linear(config["FEATURES"], config["TOKENS"])		# should bias be disabled?
		self.log_softmax = nn.LogSoftmax()
		self.apply(self._init_weights)
		print("number of parameters: {}".format(sum(p.numel() for p in self.parameters())))

	def _init_weights(self, module):
		if isinstance(module, nn.Embedding):
			module.weight.data.normal_(mean=0.0, std=0.02)

	def forward(self, srctexts, tgttexts):
		src_indicies = torch.cat([text2tensor(d, srctext).view(-1, 1) for srctext in srctexts], dim=1)
		src = self.embed(src_indicies)
		src_padding_mask = torch.eq(src_indicies, d.word2idx[PADDING]).float()
		src_padding_mask = src_padding_mask.masked_fill(src_padding_mask == 1, float('-inf')).masked_fill(src_padding_mask == 0, float(0.0))
		src_padding_mask = torch.transpose(src_padding_mask, 0, 1)

		tgt_indicies = torch.cat([text2tensor(d, tgttext).view(-1, 1) for tgttext in tgttexts], dim=1)
		tgt = self.embed(tgt_indicies)

		tgt_padding_mask = torch.eq(tgt_indicies, d.word2idx[PADDING]).float()
		tgt_padding_mask = tgt_padding_mask.masked_fill(tgt_padding_mask == 1, float('-inf')).masked_fill(tgt_padding_mask == 0, float(0.0))
		tgt_padding_mask = torch.transpose(tgt_padding_mask, 0, 1)

		# print(src.shape)
		# print(tgt.shape)
		# print(src_padding_mask.shape)
		# print(tgt_padding_mask.shape)

		output = self.transformer(src, tgt,
								  tgt_mask=self.transformer.generate_square_subsequent_mask(LEN),
								  src_key_padding_mask=src_padding_mask,
								  tgt_key_padding_mask=tgt_padding_mask).view(-1, FEATURES)
		output = self.lin_out(output)
		output = self.log_softmax(output)
		#print(output)
		# print(output.shape)
		# print(tgt_indicies.shape)

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
for epoch in range(80):
	epoch_loss = 0.
	for batch in range(BATCHES):
	    optimizer.zero_grad()
	    q_chunk = dataset_questions[BLOCKSIZE * batch:BLOCKSIZE * (batch + 1)]
	    a_chunk = dataset_answers[BLOCKSIZE * batch:BLOCKSIZE * (batch + 1)]
	    #print(q_chunk)
	    #print(a_chunk)
	    output, tgt_ref = model(q_chunk, a_chunk)
	    #print(output.shape)
	    #print(tgt.shape)
	    loss = criterion(output, tgt_ref.view(-1))
	    loss.backward()
	    optimizer.step()
	    epoch_loss += loss.item()
	print("Epcoh {}, Loss: {}".format(epoch, epoch_loss))

model.eval()
print()

training_questions = set(dataset_questions)
correct = 0
total = 200
seen = 0
for i in range(total):
	num1 = random.randint(0, 200)
	num2 = random.randint(0, 200)
	q, a = make_q_a(num1, num2)
	output, tgt_ref = model([q], [a])
	got=torch.argmax(output, 1)
	#print(got)
	#print(tgt_ref)
	output_str = tensor2text(d, tgt_ref.view(-1))
	got_str = tensor2text(d, got.view(-1))

	print("Q: {}+{} , A: {}. Seen before: {}".format(num1, num2, num1 + num2, q in training_questions))
	print("Expected: '{}'".format(output_str))
	print(colored("Got: '{}'".format(got_str), "blue" if output_str == got_str else "red"))
	print()
	correct += (output_str == got_str)
	seen += (q in training_questions)
print("{} Correct out of {} total. {:.3f}% accuracy".format(correct, total, correct/total * 100))
print("{} of the {} test questions were already seen during training".format(seen, total))
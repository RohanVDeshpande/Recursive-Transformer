import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data

SRC_TGT_SEP = u"\u2358"			# श
TGT_LOOP_SEP = u"\u2325"		# क
END = u"\u2352"					# र
PADDING = u"\u2340"				# त
LEN = 16

srctext = "5+5" + (LEN-3) * PADDING
tgttext = "10" + TGT_LOOP_SEP + "1" + END + (LEN - 5) * PADDING

print(srctext)
print(tgttext)

d = data.Dictionary()

for c in srctext:
	d.add_word(c)
for c in tgttext:
	d.add_word(c)

FEATURES = 16

#print(src_mask)
#print(tgt_mask)
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
for epoch in range(300):
    optimizer.zero_grad()
    output, tgt_ref = model(srctext, tgttext)
    #print(output.shape)
    #print(tgt.shape)
    loss = criterion(output, tgt_ref)
    loss.backward()
    optimizer.step()
    print("Loss: {}".format(loss.item()))

model.eval()
output, tgt_ref = model(srctext, tgttext)
#print(output)
#print(output.shape)
got=torch.argmax(output, 1)
print(got)
print(tgt_ref)

print("Expected: '{}'".format(tensor2text(d, tgt_ref)))
print("Got: '{}'".format(tensor2text(d, got)))
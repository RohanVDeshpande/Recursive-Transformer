import math
import random

import torch
import torch.nn as nn

class FRT(nn.Module):
	""" Forced Recursive Transformer """

	def __init__(self, config):
		super().__init__()
		for key in config:
			setattr(self, key, config[key])

		assert self.TOKENS is not None
		assert self.FEATURES is not None
		assert self.HEADS is not None
		assert self.ENC_LAYERS is not None
		assert self.DEC_LAYERS is not None
		assert self.FEED_FORWARD is not None
		assert self.TGT_LEN is not None

		self.embed = nn.Embedding(self.TOKENS, self.FEATURES)
		
		# nn.Transformer parameters:
		# d_model: int = 512,
		# nhead: int = 8,
		# num_encoder_layers: int = 6,
		# num_decoder_layers: int = 6,
		# dim_feedforward: int = 2048,
		# dropout: float = 0.1,
		# activation: str = 'relu'
		self.transformer = nn.Transformer(d_model=self.FEATURES, nhead=self.HEADS, num_encoder_layers=self.ENC_LAYERS, num_decoder_layers=self.DEC_LAYERS, dim_feedforward=self.FEED_FORWARD)
		self.lin_out = nn.Linear(self.FEATURES, self.TOKENS)		# should bias be disabled?
		self.log_softmax = nn.LogSoftmax()
		self.apply(self._init_weights)
		print("FRT model has {} parameters".format(sum(p.numel() for p in self.parameters())))

	def _init_weights(self, module):
		if isinstance(module, nn.Embedding):
			module.weight.data.normal_(mean=0.0, std=0.02)

	def forward(self, src_indicies, src_padding_mask, tgt_indicies, tgt_padding_mask):
		src = self.embed(src_indicies)
		tgt = self.embed(tgt_indicies)

		output = self.transformer(src, tgt,
								  tgt_mask=self.transformer.generate_square_subsequent_mask(self.TGT_LEN),
								  src_key_padding_mask=src_padding_mask,
								  tgt_key_padding_mask=tgt_padding_mask).view(-1, self.FEATURES)
		output = self.lin_out(output)
		output = self.log_softmax(output)

		return output, tgt_indicies


	def predict(self, src_indicies, src_padding_mask):
		src = self.embed(src_indicies)
		memory = self.transformer.encoder(src, mask=src_mask, src_key_padding_mask=src_padding_mask)

		tgt_indicies = torch.zeros(1, src_indicies.shape[1])							# (1, N)
		#tgt = torch.full((1, src_indicies.shape[1], self.FEATURES), start_token_index)	# (1, N, E)

		# tgt_key_padding_mask=???
		steps = math.round(1.5 * self.TGT_LEN)
		for k in steps:
			tgt = self.embed(tgt_indicies)		# (1, N, E)
        	output = self.transformer.decoder(tgt, memory, tgt_mask=self.transformer.generate_square_subsequent_mask(k + 1))
        	# output -> (T, N, E)

        	output = output[-1, :, :]	# (1, N, E)

        	output = self.lin_out(output)
			output = self.log_softmax(output)
			output = torch.argmax(output, 1)

			tgt_indicies[-1, :] = output
			tgt_indicies = torch.cat((tgt_indicies, torch.zeros(1, src_indicies.shape[1])))

		return tgt_indicies
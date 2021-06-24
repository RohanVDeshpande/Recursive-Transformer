import math
import random

import torch
import torch.nn as nn

from transformer_utils.PositionalEncoder import PositionalEncoding
from transformer_utils.CausalDecoder import CausalTransformerDecoder, CausalTransformerDecoderLayer

class FRT(nn.Module):
	""" Forced Recursive Transformer """

	def __init__(self, config):
		super().__init__()
		self.CAUSAL_INFERENCE=False
		for key in config:
			setattr(self, key, config[key])

		assert self.TOKENS is not None
		assert self.FEATURES is not None
		assert self.HEADS is not None
		assert self.ENC_LAYERS is not None
		assert self.DEC_LAYERS is not None
		assert self.FEED_FORWARD is not None
		assert self.SRC_LEN is not None
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

		self.transformer = nn.Transformer(d_model=self.FEATURES, nhead=self.HEADS, num_encoder_layers=self.ENC_LAYERS, num_decoder_layers=self.DEC_LAYERS, \
				 						  dim_feedforward=self.FEED_FORWARD, custom_decoder=CausalTransformerDecoder(
				 						  			CausalTransformerDecoderLayer(d_model=self.FEATURES, nhead=self.HEADS, dim_feedforward=self.FEED_FORWARD),
				 						  			self.DEC_LAYERS, torch.nn.LayerNorm(self.FEATURES)) if self.CAUSAL_INFERENCE else None
				 						  )
		self.src_pos_encoder = PositionalEncoding(self.FEATURES)
		self.tgt_pos_encoder = PositionalEncoding(self.FEATURES)
		self.lin_out = nn.Linear(self.FEATURES, self.TOKENS)		# should bias be disabled?
		self.log_softmax = nn.LogSoftmax()
		self.apply(self._init_weights)
		print("FRT model has {} parameters".format(sum(p.numel() for p in self.parameters())))

	def _init_weights(self, module):
		if isinstance(module, nn.Embedding):
			module.weight.data.normal_(mean=0.0, std=0.02)

	def gen_mask(self, sz):
	    mask = (torch.triu(torch.ones(sz, sz)) + torch.eye(sz) == 1).transpose(0, 1)
	    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
	    return mask


	def forward(self, src_indicies, tgt_indicies, src_padding_mask, tgt_padding_mask):
		src = self.embed(src_indicies)
		src = self.src_pos_encoder(src)
		tgt = self.embed(tgt_indicies)
		tgt = self.tgt_pos_encoder(tgt)
		tgt_mask = self.transformer.generate_square_subsequent_mask(self.TGT_LEN).to(self.device)

		output = self.transformer(src, tgt,
		                          tgt_mask=tgt_mask,
		                          src_key_padding_mask=src_padding_mask,
		                          tgt_key_padding_mask=tgt_padding_mask,
		                          memory_key_padding_mask=src_padding_mask)
		output = output[:-1,:, :].view(-1, self.FEATURES)
		output = self.lin_out(output)
		output = self.log_softmax(output)
		#print(output.shape)
		#print(tgt_indicies.shape)
		return output, tgt_indicies[1:, :]


	def predict(self, src_indicies, src_padding_mask, start_token_index1, start_token_index2):
		#print(src_indicies)
		src = self.embed(src_indicies)
		src = self.src_pos_encoder(src)
		memory = self.transformer.encoder(src, src_key_padding_mask=src_padding_mask)

		tgt_indicies = torch.cat(torch.full((1, src_indicies.shape[1]), start_token_index1, dtype=torch.long, device=self.device),
								 torch.full((1, src_indicies.shape[1]), start_token_index2, dtype=torch.long, device=self.device)
								)		# (2, N)

		cache = None
		# tgt_key_padding_mask=???
		steps = round(1.5 * self.TGT_LEN)
		for k in range(steps):
			tgt = self.embed(tgt_indicies)		# (1, N, E)
			tgt = self.tgt_pos_encoder(tgt)
			tgt_mask = self.transformer.generate_square_subsequent_mask(k + 1).to(self.device)
			#print(tgt_mask)
			if self.CAUSAL_INFERENCE:
				output, cache = self.transformer.decoder(tgt, memory, cache, tgt_mask=tgt_mask, memory_key_padding_mask=src_padding_mask)
			else:
				output = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=src_padding_mask)
                        # output -> (T, N, E)
			#print(output.shape)
			output = output[-1, :, :]	# (1, N, E)
			#print(output.shape)

			output = self.lin_out(output)
			#print(output.shape)
			output = self.log_softmax(output)
			#print(output)
			#print(output.shape)
			output = torch.argmax(output, 1)
			#print(output.shape)

			#tgt_indicies[-1, :] = output
			tgt_indicies = torch.cat((tgt_indicies, output.view(1, -1)))
			#print(tgt_indicies)
			#print(tgt_indicies.shape)
			#assert 0

			#assert k == 0

		return tgt_indicies

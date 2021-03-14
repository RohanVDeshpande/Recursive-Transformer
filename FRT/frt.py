import math
import random

import torch
import torch.nn as nn
from CausalDecoder import CausalTransformerDecoder, CausalTransformerDecoderLayer

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


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


	def predict(self, src_indicies, src_padding_mask, start_token_index):
		#print(src_indicies)
		src = self.embed(src_indicies)
		src = self.src_pos_encoder(src)
		memory = self.transformer.encoder(src, src_key_padding_mask=src_padding_mask)

		tgt_indicies = torch.full((1, src_indicies.shape[1]), start_token_index, dtype=torch.long, device=self.device)		# (1, N)
		#print(tgt_indicies.shape)
		#tgt = torch.full((1, src_indicies.shape[1], self.FEATURES), start_token_index)	# (1, N, E)
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
	
	def predict_recursive(self, src_indicies, src_padding_mask, start_token_index, loop_cont_index, padding_index):
		"""
		Runtime flags: 	--model-config config/frt_2-2.json 
						--params checkpoints/FRT_2.2/FRT_2.2_epoch2_terminated.pt  
						--dict output/dict/FRT_2.2_dict.json 
						--mode test 
						--data=deepmind_dataset/arithmetic__add_sub_multiple.tsv 
						--dict=output/dict/FRT_2.2_dict.json
		NOTE: Path to --data may be different.
		"""
		# First predict() call outside of while loop to get target length T to initialize output - messy implementation
		tgt_indicies = self.predict(src_indicies, src_padding_mask, start_token_index)	
		
		# Dimension: N x T - transposed to T x N at the end
		output = torch.zeros(tgt_indicies.shape[1], tgt_indicies.shape[0]).int().type(torch.LongTensor)
		finished = []						# Indicies of samples that have LOOP_STOP
		N = output.shape[0]					# Batch size

		while(True):
			tgt_indicies_trans = torch.transpose(tgt_indicies, 0, 1)			# N x T - for easier indexing by batch
			cont = torch.eq(tgt_indicies_trans, float(loop_cont_index))	
			mask = torch.tensor([1 if True in row else 0 for row in cont])

			if not any(mask):				# mask contains all 0s
				break

			cont_indicies = torch.squeeze(torch.nonzero(mask), 1)				# Samples with LOOP_CONT
			discont_indicies = [i for i in range(len(mask)) if mask[i] == 0]	# Samples with LOOP_STOP

			for ind in discont_indicies:	# Add new finished samples to output - worried about being slow here
				if ind not in finished:
					output[ind] = tgt_indicies_trans[ind]
					finished.append(ind)

			src_indicies = tgt_indicies
			src_padding_mask = torch.eq(tgt_indicies_trans, padding_index)		# TODO: Verify if this is correct

			if len(finished) == N:			# All samples finished, avoid calling predict() again
				break

			tgt_indicies = self.predict(src_indicies, src_padding_mask, start_token_index)

		output = torch.transpose(output, 0, 1)
		return output
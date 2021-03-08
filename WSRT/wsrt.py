import math
import random

import torch
import torch.nn as nn


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


class WSRT(nn.Module):
	""" Weakly Supervised Recursive Transformer """

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
		self.transformer = nn.Transformer(d_model=self.FEATURES, nhead=self.HEADS, num_encoder_layers=self.ENC_LAYERS, num_decoder_layers=self.DEC_LAYERS, dim_feedforward=self.FEED_FORWARD)
		self.pos_encoder = PositionalEncoding(self.FEATURES)
		self.lin_out = nn.Linear(self.FEATURES, self.TOKENS)		# should bias be disabled?
		self.log_softmax = nn.LogSoftmax()
		self.apply(self._init_weights)
		print("FRT model has {} parameters".format(sum(p.numel() for p in self.parameters())))

	def _init_weights(self, module):
		if isinstance(module, nn.Embedding):
			module.weight.data.normal_(mean=0.0, std=0.02)


	def forward(self, indicies, numSteps, start_token_index, end_token_index, hidden_length):
		print("Start idx: {}, End idx: {}".format(start_token_index, end_token_index))
		for i in range(numSteps):
			print(i)
			# pad everything after the end token:
			src_padding_mask = torch.cat([torch.tensor([ 0 if len((indicies[:, i] == end_token_index).nonzero(as_tuple=True)[0]) == 0 or \
															j <= (indicies[:, i] == end_token_index).nonzero(as_tuple=True)[0][0] else 1 \
													for j in range(indicies.shape[0])], dtype=torch.long).view(1, -1)
											for i in range(indicies.shape[1])], dim=0)

			indicies =  self.predictOneStep(indicies, src_padding_mask, start_token_index, hidden_length, i == numSteps - 1)
		return indicies


	def predictOneStep(self, src_indicies, src_padding_mask, start_token_index, hidden_length, lastStep):
		#print(src_indicies)
		src = self.embed(src_indicies)
		src = self.pos_encoder(src)
		memory = self.transformer.encoder(src, src_key_padding_mask=src_padding_mask)

		logits = []
		tgt_indicies = torch.full((1, src_indicies.shape[1]), start_token_index, dtype=torch.long, device=self.device)		# (1, N)

		for k in range(hidden_length):
			tgt = self.embed(tgt_indicies)		# (1, N, E)
			tgt = self.pos_encoder(tgt)
			tgt_mask = self.transformer.generate_square_subsequent_mask(k + 1).to(self.device)
			#print(tgt_mask)
			output = self.transformer.decoder(tgt, memory, tgt_mask=tgt_mask) # (T, N, E)
			output = output[-1, :, :]	# (1, N, E)
			output = self.lin_out(output)
			output = self.log_softmax(output)
			if lastStep:
				logits.append(output)
			
			output = torch.argmax(output, 1)
			tgt_indicies = torch.cat((tgt_indicies, output.view(1, -1)))
			
		if lastStep:
			return torch.cat(logits).view(-1, self.TOKENS)
		else:
			return tgt_indicies
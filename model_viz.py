import math
import random
from termcolor import colored
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import argparse

from matplotlib import pyplot as plt
import seaborn

from FRT import utils
from FRT import data
from FRT import frt
from WSRT import wsrt

parser = argparse.ArgumentParser(description='Forced Recursive Transformer')
parser.add_argument('--type', type=str, required=True,
                    help="Model Type",
					choices=["frt", "wsrt"])
parser.add_argument('--model-config', type=str, required=True,
                    help='Model json config')
parser.add_argument('--params', type=str, required=True, help='Model path')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')


args = parser.parse_args()

assert os.path.isdir("output/"), "You need a folder called 'output' in order to save model data / test predictions"
assert os.path.isdir("tb/"), "You need a folder called 'tb' for tensorboard"
assert os.path.isdir("output/dict/"), "You need a folder called 'dict' in order to save/load a dictionary"

# Set the random seed manually for reproducibility.
# torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
    	utils.confirm("You have a CUDA device, so you should probably run with --cuda.")
    else:
    	print("Using CUDA device")

device = torch.device("cuda" if args.cuda else "cpu")


with open(args.model_config) as f:
  model_config = json.load(f)
  dictionary_path = "output/dict/{}_dict.json".format(model_config["NAME"])

  with open(model_config["DATASET_CONFIG"]) as f:
  	dataset_config = json.load(f)

dataset = data.Dataset(dataset_config)
print("Loading dictionary from: {}".format(dictionary_path))
dataset.loadDictionary(dictionary_path)
dataset.device = device


model_config["TOKENS"] = dataset.tokens()
model_config["SRC_LEN"] = dataset.SRC_LEN
model_config["TGT_LEN"] = dataset.TGT_LEN

if args.type == "frt":
	model = frt.FRT(model_config)
elif args.type == "wsrt":
	model = wsrt.WSRT(model_config)
model.device = device
model.to(device)

model.load_state_dict(torch.load(args.params, map_location=torch.device(device)), strict=False)
model.eval()


fig, axs = plt.subplots(1, model.ENC_LAYERS)


srctext = "1+2+3+4"
srctext = dataset.START + srctext + dataset.END
if args.type == "wsrt":
	srctext += (dataset.TGT_LEN - len(srctext)) * dataset.PADDING
src_indicies = dataset.text2tensor(srctext).view(-1, 1)

print(src_indicies)

srclabel = srctext.replace(dataset.START, "S").replace(dataset.END, "E").replace(dataset.PADDING, "P")

with torch.no_grad():
	def run_encoder_layer(layer, src, src_key_padding_mask):
	    
	    src2, src_attn = layer.self_attn(src, src, src, attn_mask=None,
	                          key_padding_mask=src_key_padding_mask)
	    src = src + layer.dropout1(src2)
	    src = layer.norm1(src)
	    src2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(src))))
	    src = src + layer.dropout2(src2)
	    src = layer.norm2(src)
	    return src, src_attn


	def run_encoder (encoder, src, src_key_padding_mask):
	    output = src
	    attentions = []
	    for mod in encoder.layers:
	        output, attn = run_encoder_layer(mod, output, src_key_padding_mask)
	        attentions.append(attn)

	    if encoder.norm is not None:
	        output = encoder.norm(output)

	    return output, attentions



	def run_forward(model, src_indicies, src_key_padding_mask):
		src = model.embed(src_indicies)
		if args.type == "frt":
			src = model.src_pos_encoder(src)
		elif args.type == "wsrt":
			src = model.pos_encoder(src)
		memory, encoder_attentions = run_encoder(model.transformer.encoder, src, src_key_padding_mask)

		print(encoder_attentions[0])
		print(encoder_attentions[0].shape)

		return encoder_attentions


	encoder_attentions = run_forward(model, src_indicies, None)

	for i in range(len(encoder_attentions)):
		seaborn.heatmap(torch.squeeze(encoder_attentions[i]).detach().numpy(), xticklabels=srclabel, yticklabels=srclabel, vmin=0.0, vmax=1.0, cbar=False, ax=axs[i], square=True)

	fig.set_size_inches(15,5)
	fig.savefig("output/{}_attention_viz.png".format(model_config["NAME"]))

# tgt_sent = trans.split()
# def draw(data, x, y, ax):
#     seaborn.heatmap(data, 
#                     xticklabels=x, square=True, yticklabels=y, vmin=0.0, vmax=1.0, 
#                     cbar=False, ax=ax)
    
# for layer in range(1, 6, 2):
#     fig, axs = plt.subplots(1,4, figsize=(20, 10))
#     print("Encoder Layer", layer+1)
#     for h in range(4):
#         draw(model.encoder.layers[layer].self_attn.attn[0, h].data, 
#             sent, sent if h ==0 else [], ax=axs[h])
#     plt.show()
    
# for layer in range(1, 6, 2):
#     fig, axs = plt.subplots(1,4, figsize=(20, 10))
#     print("Decoder Self Layer", layer+1)
#     for h in range(4):
#         draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(tgt_sent)], 
#             tgt_sent, tgt_sent if h ==0 else [], ax=axs[h])
#     plt.show()
#     print("Decoder Src Layer", layer+1)
#     fig, axs = plt.subplots(1,4, figsize=(20, 10))
#     for h in range(4):
#         draw(model.decoder.layers[layer].self_attn.attn[0, h].data[:len(tgt_sent), :len(sent)], 
#             sent, tgt_sent if h ==0 else [], ax=axs[h])
#     plt.show()
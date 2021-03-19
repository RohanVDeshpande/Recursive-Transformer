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

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

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
parser.add_argument('--dims', type=int, default=3,
                    help="Number of dimensions",
					choices=[2, 3])
parser.add_argument('--num-only', action='store_true',
                    help='Only consider numerical characters')
parser.add_argument('--gif', action='store_true',
                    help='Create a gif')


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


X = model.embed.weight.detach().numpy()
X = X[:len(dataset.dictionary.idx2word), :]
print(X.shape)

if args.num_only:
	print("Pruning dictionary & embedding space to only include numbers")
	new_idx2word = []
	for j in range(len(dataset.dictionary.idx2word) - 1, 0, -1):
		if dataset.dictionary.idx2word[j].isdigit():
			new_idx2word = [dataset.dictionary.idx2word[j]] + new_idx2word
		else:
			X = np.vstack((X[:j, :], X[j+1:,:]))
	#print(X)
	#print(X.shape)
	print(new_idx2word)
	dataset.dictionary.idx2word = new_idx2word
pca = PCA(n_components=args.dims, whiten=True)
pca.fit(X)
X_pca = pca.transform(X)
#print(X_pca.shape)

print(X_pca)

if args.dims == 2:
	fig = plt.figure()
	ax = fig.add_subplot(111)

	delta = 0.015

	for i in range(len(dataset.dictionary.idx2word)):
		ax.scatter(X_pca[i, 0], X_pca[i, 1])
		if dataset.dictionary.idx2word[i] == dataset.PADDING:
			ax.text(X_pca[i, 0] + delta, X_pca[i, 1] + delta, "PADDING")
		elif dataset.dictionary.idx2word[i] == dataset.START:
			ax.text(X_pca[i, 0] + delta, X_pca[i, 1] + delta, "START")
		elif dataset.dictionary.idx2word[i] == dataset.END:
			ax.text(X_pca[i, 0] + delta, X_pca[i, 1] + delta, "END")
		elif dataset.dictionary.idx2word[i] == dataset.TGT_LOOP_SEP:
			ax.text(X_pca[i, 0] + delta, X_pca[i, 1] + delta, "TGT_LOOP_SEP")
		elif dataset.dictionary.idx2word[i] == dataset.LOOP_CONTINUE:
			ax.text(X_pca[i, 0] + delta, X_pca[i, 1] + delta, "LOOP_CONTINUE")
		elif dataset.dictionary.idx2word[i] == dataset.LOOP_STOP:
			ax.text(X_pca[i, 0] + delta, X_pca[i, 1] + delta, "LOOP_STOP")
		else:
			ax.text(X_pca[i, 0] + delta, X_pca[i, 1] + delta, dataset.dictionary.idx2word[i])
	fig.set_size_inches(8,6)
	fig.savefig("output/{}_embedding_pca_2d{}.png".format(model_config["NAME"], "_numonly" if args.num_only else ""))
elif args.dims == 3:
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	delta = 0.015

	for i in range(len(dataset.dictionary.idx2word)):
		ax.scatter(X_pca[i, 0], X_pca[i, 1], X_pca[i, 2])
		if dataset.dictionary.idx2word[i] == dataset.PADDING:
			ax.text(X_pca[i, 0] + delta, X_pca[i, 1] + delta, X_pca[i, 2] + delta, "PADDING")
		elif dataset.dictionary.idx2word[i] == dataset.START:
			ax.text(X_pca[i, 0] + delta, X_pca[i, 1] + delta, X_pca[i, 2] + delta, "START")
		elif dataset.dictionary.idx2word[i] == dataset.END:
			ax.text(X_pca[i, 0] + delta, X_pca[i, 1] + delta, X_pca[i, 2] + delta, "END")
		elif dataset.dictionary.idx2word[i] == dataset.TGT_LOOP_SEP:
			ax.text(X_pca[i, 0] + delta, X_pca[i, 1] + delta, X_pca[i, 2] + delta, "TGT_LOOP_SEP")
		elif dataset.dictionary.idx2word[i] == dataset.LOOP_CONTINUE:
			ax.text(X_pca[i, 0] + delta, X_pca[i, 1] + delta, X_pca[i, 2] + delta, "LOOP_CONTINUE")
		elif dataset.dictionary.idx2word[i] == dataset.LOOP_STOP:
			ax.text(X_pca[i, 0] + delta, X_pca[i, 1] + delta, X_pca[i, 2] + delta, "LOOP_STOP")
		else:
			ax.text(X_pca[i, 0] + delta, X_pca[i, 1] + delta, X_pca[i, 2] + delta, dataset.dictionary.idx2word[i])
	fig.set_size_inches(8,6)


	def update(angle):
	    print("timestep {}".format(angle))
	    ax.view_init(30, angle)
	    return ax


	if args.gif:
		anim = FuncAnimation(fig, update, frames=range(0, 360), interval=150)
		anim.save("output/{}_embedding_pca_3d{}.gif".format(model_config["NAME"], "_numonly" if args.num_only else ""), dpi=80, writer='imagemagick')
	else:
		fig.savefig("output/{}_embedding_pca_3d{}.png".format(model_config["NAME"], "_numonly" if args.num_only else ""))
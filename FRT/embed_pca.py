import math
import random
from termcolor import colored
import json
import utils
import os
import sys

import torch
import torch.nn as nn
import argparse
import frt

import data

from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation


parser = argparse.ArgumentParser(description='Forced Recursive Transformer')
parser.add_argument('--model-config', type=str, required=True,
                    help='Model json config')
parser.add_argument('--params', type=str, required=True, help='Model path')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')

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
  dictionary_path = "output/dict/{}_dict.json".format(os.path.splitext(os.path.basename(args.params))[0])

  with open(model_config["DATASET_CONFIG"]) as f:
  	dataset_config = json.load(f)

dataset = data.Dataset(dataset_config)
print("Loading dictionary from: {}".format(dictionary_path))
dataset.loadDictionary(dictionary_path)
dataset.device = device


model_config["TOKENS"] = dataset.tokens()
model_config["SRC_LEN"] = dataset.SRC_LEN
model_config["TGT_LEN"] = dataset.TGT_LEN

model = frt.FRT(model_config)
model.device = device
model.to(device)

model.load_state_dict(torch.load(args.params, map_location=torch.device(device)))
model.eval()


X = model.embed.weight.detach().numpy()

pca = PCA(n_components=3, whiten=True)
pca.fit(X)
X_pca = pca.transform(X)
print(X_pca.shape)

print(X_pca)

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
	else:
		ax.text(X_pca[i, 0] + delta, X_pca[i, 1] + delta, X_pca[i, 2] + delta, dataset.dictionary.idx2word[i])
fig.set_size_inches(8,6)


def update(angle):
    print("timestep {}".format(angle))
    ax.view_init(30, angle)
    return ax


if args.gif is not None:
	anim = FuncAnimation(fig, update, frames=range(0, 360), interval=150)
	anim.save('output/embedding_pca.gif', dpi=80, writer='imagemagick')
else:
	fig.savefig("output/embedding_pca.png")
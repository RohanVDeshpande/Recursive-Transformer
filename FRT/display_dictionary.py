import json
import utils
import argparse
import data


parser = argparse.ArgumentParser(description='Display and Save Dictionaries built from dataset')
parser.add_argument('--data', type=str, required=True,
                    help='Dataset path')
parser.add_argument('--dataset-config', type=str, required=True,
                    help='Dataset configuration path')
parser.add_argument('--save', type=str,
                    help='Dataset path')

args = parser.parse_args()

with open(args.dataset_config) as f:
  	dataset_config = json.load(f)
dataset = data.Dataset(dataset_config)
dataset.buildDataset(args.data)
print(dataset.dictionary.idx2word)
print(dataset.dictionary.word2idx)
if args.save:
	dataset.saveDictionary(args.save)
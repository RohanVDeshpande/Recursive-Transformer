import os
from io import open
import torch
import math
import copy
from torch.utils.data import Dataset, DataLoader
import json
import random

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.freeze_dict = False

    def add_word(self, word):
        if word not in self.word2idx:
            if self.freeze_dict:
                assert 0, "'{}' was not already in Dictionary object!".format(word)
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def len(self):
        return len(self.idx2word)


class Dataset(Dataset):
    def __init__(self, config):
        self.questions = []
        self.answers = []
        self.steps = []
        self.types = []
        self.sources = []
        self.device = "cpu"     # default device
        self.TOTAL_TOKENS = None    # manually set number of tokens

        if isinstance(config, dict):
            self.dictionary = Dictionary()
            for key in config:
                setattr(self, key, config[key])
        else:
            self.dictionary = copy.deepcopy(config.dictionary)
            self.dictionary.freeze_dict = True
            print("Using existing dictionary to initialize new dataset object", self.dictionary.idx2word)
            self.START = config.START
            self.SRC_LEN = config.SRC_LEN
            self.TGT_LEN = config.TGT_LEN
            self.PADDING = config.PADDING
            self.END = config.END
            self.TGT_LOOP_SEP = config.TGT_LOOP_SEP
            self.LOOP_CONTINUE = config.LOOP_CONTINUE
            self.LOOP_STOP = config.LOOP_STOP
            self.BATCH_SIZE = config.BATCH_SIZE
            self.TOTAL_TOKENS = config.TOTAL_TOKENS

        assert self.START is not None
        assert self.SRC_LEN is not None
        assert self.TGT_LEN is not None
        assert self.PADDING is not None
        assert self.END is not None
        assert self.TGT_LOOP_SEP is not None
        assert self.LOOP_CONTINUE is not None
        assert self.LOOP_STOP is not None
        assert self.BATCH_SIZE is not None

    def tokenizeQuestion(self, q):
        q = self.START + q + self.END
        assert self.SRC_LEN - len(q) >= 0, "Q length is {} but SRC_LEN={}".format(len(q), self.SRC_LEN)
        q += (self.SRC_LEN - len(q)) * self.PADDING
        return q

    def tokenizeAnswer(self, a):
        a = self.START + a + self.END
        assert self.TGT_LEN - len(a) >= 0, "A length is {} but TGT_LEN={}".format(len(a), self.TGT_LEN)
        a += (self.TGT_LEN - len(a)) * self.PADDING
        return a

    def populateDictionary(self, text):
        for c in text:
            self.dictionary.add_word(c)

    def tokens(self):
        if self.TOTAL_TOKENS is not None:
            assert self.TOTAL_TOKENS >= len(self.dictionary.idx2word), "Manually defined token count should be at least same as dictionary token count"
            return self.TOTAL_TOKENS
        else:
            return len(self.dictionary.idx2word)

    def loadDictionary(self, dict_path):
        with open(dict_path, "r") as f:
            json_dict = json.load(f)
            self.dictionary.idx2word = [-1] * len(json_dict)
            for key in json_dict:
                self.dictionary.idx2word[json_dict[key]] = key
                self.dictionary.word2idx[key] = json_dict[key]
        self.dictionary.freeze_dict = True

    def saveDictionary(self, dict_path):
        with open(dict_path, "w") as f:
            json.dump(self.dictionary.word2idx, f) 

    def buildDataset(self, path_or_array):
        if isinstance(path_or_array, str):
            assert os.path.exists(path_or_array)
            self.sources.append((path_or_array, self.__len__()))
            with open(path_or_array, 'r', encoding="utf8") as f:
                for line in f:
                    l = line.strip().split('\t')
                    # print(f'l = {l}')
                    q, a, = self.tokenizeQuestion(l[0]), self.tokenizeAnswer(l[1])
                    self.questions.append(q)
                    self.answers.append(a)
                    self.steps.append(int(l[2]))
                    #self.types.append(l[3])        # unneeded right now

                    self.populateDictionary(q)
                    self.populateDictionary(a)
        else:
            for line in path_or_array:
                l = line.strip().split('\t')
                # print(f'l = {l}')
                q, a, = self.tokenizeQuestion(l[0]), self.tokenizeAnswer(l[1], l[2])
                self.questions.append(q)
                self.answers.append(a)
                #self.types.append(l[3])        # unneeded right now

                self.populateDictionary(q)
                self.populateDictionary(a)   

    def __len__(self):
        return len(self.questions)


    def text2tensor(self, text):
        return torch.tensor([ self.dictionary.word2idx[c] for c in text], dtype=torch.long)


    # t -> (T, N)

    def tensor2text(self, t):
        if len(t.shape) == 1:
            return "".join([self.dictionary.idx2word[idx] for idx in t])
        elif len(t.shape) == 2:
            return [ "".join([self.dictionary.idx2word[t[j][i]] for j in range(t.shape[0])]) for i in range(t.shape[1])]
        else:
            assert 0        # not implemented yet

    def __getitem__(self, idx):
        src_indicies = self.text2tensor(self.questions[idx]).view(-1, 1)
        #print(src_indicies.shape)
        tgt_indicies = self.text2tensor(self.answers[idx]).view(-1, 1)
        return src_indicies, tgt_indicies, self.steps[idx]


def dataset_collate_fn(batch):
    src_indicies, tgt_indicies, steps = zip(*batch)

    if random.random() < 0.4:
        src_indicies = torch.cat(src_indicies, dim=1)
        tgt_indicies = torch.cat(tgt_indicies, dim=1)
        steps = max(steps)
    else:
        split = {}
        for i in range(len(steps)):
            if steps[i] not in split:
                split[steps[i]] = [[],[], 0]
            split[steps[i]][0].append(src_indicies[i])
            split[steps[i]][1].append(tgt_indicies[i])
            split[steps[i]][2] += 1
        more_frequent_step = None
        step_frequency = 0
        for key in split:
            if split[key][2] >= step_frequency:
                more_frequent_step = key
                step_frequency = split[key][2]
        assert more_frequent_step is not None, "Couldn't find most frequent step"
        # print(more_frequent_step)
        # print(split[more_frequent_step][0])
        src_indicies = torch.cat(split[more_frequent_step][0], dim=1)
        tgt_indicies = torch.cat(split[more_frequent_step][1], dim=1)
        steps = more_frequent_step
    
    return src_indicies, tgt_indicies, steps

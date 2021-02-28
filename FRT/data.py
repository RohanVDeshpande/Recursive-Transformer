import os
from io import open
import torch
import math
import copy

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


class Dataset(object):
    def __init__(self, config):
        self.questions = []
        self.answers = []
        self.types = []
        self.sources = []

        if isinstance(config, dict):
            self.dictionary = Dictionary()
            for key in config:
                setattr(self, key, config[key])
        else:
            self.dictionary = copy.deepcopy(config.dictionary)
            self.dictionary.freeze_dict = True
            print(self.dictionary.idx2word)
            self.SRC_LEN = config.SRC_LEN
            self.TGT_LEN = config.TGT_LEN
            self.PADDING = config.PADDING
            self.END = config.END
            self.TGT_LOOP_SEP = config.TGT_LOOP_SEP
            self.BATCH_SIZE = config.BATCH_SIZE

        assert self.SRC_LEN is not None
        assert self.TGT_LEN is not None
        assert self.PADDING is not None
        assert self.END is not None
        assert self.TGT_LOOP_SEP is not None
        assert self.BATCH_SIZE is not None

    def tokenizeQuestion(self, q):
        #q = self.START + q
        assert self.SRC_LEN - len(q) >= 0
        q += (self.SRC_LEN - len(q)) * self.PADDING
        return q

    def tokenizeAnswer(self, a, done):
        #a = self.START + a
        a += self.TGT_LOOP_SEP + done + self.END            # hardcoded loop done. change this later
        assert self.TGT_LEN - len(a) >= 0
        a += (self.TGT_LEN - len(a)) * self.PADDING
        return a

    def populateDictionary(self, text):
        for c in text:
            self.dictionary.add_word(c)

    def tokens(self):
        return len(self.dictionary.idx2word)

    def buildDataset(self, path):
        assert os.path.exists(path)
        self.sources.append((path, self.len()))
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                l = line.strip().split('\t')
                q, a, = self.tokenizeQuestion(l[0]), self.tokenizeAnswer(l[1], l[2])
                self.questions.append(q)
                self.answers.append(a)
                self.types.append(l[3])

                self.populateDictionary(q)
                self.populateDictionary(a)

    def len(self):
        return len(self.questions)

    def batches(self):
        return math.ceil(self.len() / self.BATCH_SIZE)

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

    def get_data(self, batch_num):
        src_indicies = torch.cat([self.text2tensor(srctext).view(-1, 1) \
                                for srctext in self.questions[self.BATCH_SIZE * batch_num : self.BATCH_SIZE * (batch_num + 1)]], dim=1)

        src_padding_mask = torch.eq(src_indicies, self.dictionary.word2idx[self.PADDING]).float()
        src_padding_mask = src_padding_mask.masked_fill(src_padding_mask == 1, float('-inf')).masked_fill(src_padding_mask == 0, float(0.0))
        src_padding_mask = torch.transpose(src_padding_mask, 0, 1)


        tgt_indicies = torch.cat([self.text2tensor(tgttext).view(-1, 1) \
                                for tgttext in self.answers[self.BATCH_SIZE * batch_num : self.BATCH_SIZE * (batch_num + 1)]], dim=1)

        tgt_padding_mask = torch.eq(tgt_indicies, self.dictionary.word2idx[self.PADDING]).float()
        tgt_padding_mask = tgt_padding_mask.masked_fill(tgt_padding_mask == 1, float('-inf')).masked_fill(tgt_padding_mask == 0, float(0.0))
        tgt_padding_mask = torch.transpose(tgt_padding_mask, 0, 1)

        return src_indicies, src_padding_mask, tgt_indicies, tgt_padding_mask
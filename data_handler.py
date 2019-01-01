from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pickle
import torch as tc
from shared import PAD, UNK, START, END, LABELS, snli_train, vocab_file, EM_DIM


class SNLI(Dataset):
    def __init__(self,data, pre_trained=False):
        self.C2I = {chr(i): i for i in range(128)}
        self.C2I[PAD] = 128
        if pre_trained:
            self.word_embedded, self.index2word, self.E = self.load()
        else:
            self.word_embedded, self.index2word, self.E = None, None, None
        self.label2index = {l:i for i, l in enumerate(LABELS.values())}
        self.index2label = {v:k for k,v in self.label2index.items()}
        self.read_data(data)
        self.data_legth = len(self.data)

    def __len__(self):
        return self.data_legth

    def load_word_vocabulary(self, vocab):
        self.word_embedded, self.index2word = vocab

    def read_data(self,data):
        self.data = []
        data.readline()
        for line in data:
            label, _, _, _, _, premise, hypothesis, _, _, _, _, _, _, _ = line.split("\t")
            if label == "-":
                continue
            self.data.append((premise.split(), hypothesis.split(), label))

    def load(self, src=None):
        # load pickle if exists
        if src is None:
            pkl_path = vocab_file.strip(".txt") + ".pkl"
            src = pkl_path
        if os.path.exists(src):
            return pickle.load(open(src, "rb"))

        words = [START, END, PAD, UNK]
        mat = [np.zeros(EM_DIM), np.zeros(EM_DIM), np.zeros(EM_DIM), np.zeros(EM_DIM)]
        fd = open(vocab_file, "rt", encoding="utf-8")
        for line in fd:
            word, vec = line.split(" ", 1)
            mat.append(np.fromstring(vec, sep=" "))  # append pre trained vector
            words.append(word)  # append word
        mx = np.vstack(mat)  # concat vectors

        W2I = {word: i for i, word in enumerate(words)}
        # save as pickle
        pickle.dump((W2I, words, mx), open(src, "wb"))
        return W2I, words, mx

    def embedded_sentence(self, sentence):
        words_after_embedding = []
        for i, word in enumerate(sentence):
            words_after_embedding.append(self.word_embedded.get(word.lower(), self.word_embedded[UNK]))

        char_after_embedding= []
        max_len_char = 0
        for i, word in enumerate(sentence):
            embed_i = []
            for c in word:
                embed_i.append(self.C2I[c])
            char_after_embedding.append(embed_i)
            max_len_char = len(embed_i) if len(embed_i) > max_len_char else max_len_char

        return words_after_embedding, char_after_embedding, max_len_char

    def __getitem__(self, i):
        premise, hypothesis, label = self.data[i]
        pw_embedded, pc_embedded, pw_max_len = self.embedded_sentence(premise)
        hw_embedded, hc_embedded, hw_max_len = self.embedded_sentence(hypothesis)
        return premise, hypothesis, len(premise), len(hypothesis), pw_embedded, pc_embedded, pw_max_len, \
               hw_embedded, hc_embedded, hw_max_len, self.label2index[label]

    def collate_fn(self, batch):
        premise_words, premise_chars, hypothesis_words, hypothesis_chars, labels = [], [], [], [], []
        for sample in batch:
            premise_words.append(sample[2])
            premise_chars.append(sample[6])
            hypothesis_words.append(sample[3])
            hypothesis_chars.append(sample[9])
            labels.append(sample[10])
        max_premise_words = np.max(premise_words)
        max_premise_chrs = np.max(premise_chars)
        max_hypothesis_words = np.max(hypothesis_words)
        max_hypothesis_chrs = np.max(hypothesis_chars)

        pb, hb, pwb, hwb, pcb, hcb = [], [], [], [], [], []
        for sample in batch:
            pb.append(sample[0])
            hb.append(sample[1])
            pwb.append(sample[4] + [self.word_embedded[PAD]] * (max_premise_words - len(sample[4])))
            hwb.append(sample[7] + [self.word_embedded[PAD]] * (max_hypothesis_words - len(sample[7])))
            temp = [[self.C2I[PAD]] * (max_premise_chrs - len(chars)) + chars for chars in sample[5]]
            pcb.append([[self.C2I[PAD]] * max_premise_chrs] * (max_premise_words - len(sample[4])) + temp)
            temp = [[self.C2I[PAD]] * (max_hypothesis_chrs - len(chars)) + chars for chars in sample[8]]
            hcb.append([[self.C2I[PAD]] * max_hypothesis_chrs] * (max_hypothesis_words - len(sample[7])) + temp)

        return pb, hb, tc.Tensor(pwb).long(), tc.Tensor(hwb).long(), tc.Tensor(pcb).long(), tc.Tensor(hcb).long(),\
               tc.Tensor(labels).long()


if __name__ == '__main__':
    dataset = SNLI(open(snli_train), True)
    dataloader = DataLoader(dataset=dataset, batch_size=64, collate_fn=dataset.collate_fn)
    for i, (p, h, pw, hw, pc, wc, label) in enumerate(dataloader):
        print(i, p, pw, pc)
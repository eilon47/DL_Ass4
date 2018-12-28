import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def rnn_bilstm(lstm, sentences, lengths):
	batch = sentences.size(1)
	shape = lstm.num_layer * 2, batch, lstm.hidden_size
	h = Variable(sentences.data.new(*shape).zero_())
	c = h
	pass
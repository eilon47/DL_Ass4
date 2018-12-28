import datetime
import os

import torch
import torch.nn as nn
from numpy.ma import sqrt
from tqdm import tqdm

import utils as ut

VOCAB_SIZE = 10
IN_DIM = 300
MLP_DIM = 800
DROPOUT = 0.1
WORD_SIZE = 60
HIDDEN_SIZE = 600
LABELS_SIZE = 3
EPOCHS = 3
MOD_DICT = {0:10000, 1:5000, 2:1000}


class ResidualStackedEncoders(nn.Module):
	def __init__(self, vocab_size=VOCAB_SIZE, in_dim=IN_DIM, mlp_dim=MLP_DIM, h1=HIDDEN_SIZE, h2=HIDDEN_SIZE,
				 h3=HIDDEN_SIZE, do_r=DROPOUT, word_size=WORD_SIZE, labels_size=LABELS_SIZE):
		super(ResidualStackedEncoders, self).__init__()
		self.vocab_size = vocab_size
		self.in_dim = IN_DIM
		self.mlp_dim = mlp_dim
		self.in_dim = in_dim
		self.E = nn.Embedding(vocab_size, in_dim)
		self.bi_lstm1 = nn.LSTM(input_size=in_dim, hidden_size=h1, num_layers=1,bidirectional=True)
		self.bi_lstm2 = nn.LSTM(input_size=(in_dim + h1*2), hidden_size=h2, num_layers=1,bidirectional=True)
		self.bi_lstm3 = nn.LSTM(input_size=(in_dim + h1*2), hidden_size=h3, num_layers=1,bidirectional=True)
		self.word_size = word_size
		self.dropout_rate = do_r
		self.hidden_size = [h1, h2, h3]
		self.mlp = nn.Linear(h3*8, mlp_dim)
		self.softmax = nn.Linear(mlp_dim, labels_size)
		# TODO Possible to change Tanh to ReLU
		self.classifier = nn.Sequential(*[self.mlp, nn.Tanh(), nn.Dropout(do_r), self.softmax])

	def forward(self, sentence1, label1, sentence2, label2):
		label1 = label1.clamp(max=self.word_size) 	# Convert the label to Tensor in max length of word_size
		label2 = label2.clamp(max=self.word_size)
		if sentence1.size(0) > self.word_size: 	# Adjusting the sentence to out length.
			sentence1 = sentence1[:self.word_size, :]
		if sentence2.size(0) > self.word_size:
			sentence2 = sentence2[:self.word_size, :]

		encode1_1 = self.E(sentence1) 	# Encode the sentence to its vector.
		encode2_1 = self.E(sentence2)

		s1_output1 = ut.rnn_bilstm(self.bi_lstm1, encode1_1, label1)
		s2_output1 = ut.rnn_bilstm(self.bi_lstm1, encode2_1, label2)

		# truncate
		l1 = s1_output1.size(0)
		l2 = s2_output1.size(0)
		encode1_1 = encode1_1[:l1, :, :]
		encode2_1 = encode2_1[:l2, :, :]

		s1_input_2 = torch.cat([encode1_1, s1_output1], dim=2)
		s2_input_2 = torch.cat([encode2_1, s2_output1], dim=2)

		s1_output2 = ut.rnn_bilstm(self.bi_lstm2, s1_input_2, label1)
		s2_output2 = ut.rnn_bilstm(self.bi_lstm2, s2_input_2, label2)

		s1_input_3 = torch.cat([encode1_1, s1_output1 + s1_output2], dim=2)
		s2_input_3 = torch.cat([encode2_1, s2_output1 + s2_output2], dim=2)

		s1_output3 = ut.rnn_bilstm(self.bi_lstm3, s1_input_3, label1)
		s2_output3 = ut.rnn_bilstm(self.bi_lstm3, s2_input_3, label2)

		s1_output3 = ut.max_time(s1_output3, label1)
		s2_output3 = ut.max_time(s2_output3, label2)

		mlp_input = torch.cat([s1_output3, s2_output3, torch.abs(s1_output3 - s2_output3), s1_output3 * s2_output3 ], dim=1)
		return self.classifier(mlp_input)




class Trainer:
	def __init__(self, seed=12, rate=DROPOUT, mlp_dim=MLP_DIM):
		self.mlp_dim = mlp_dim
		self.do_r = rate
		self.seed = seed
		# TODO LAYERS ?

	def train(self):
		torch.manual_seed(self.seed)
		#torch.cuda.manual_seed(self.seed)  ## TODO Maybe there is no cuda

		snli_data, mlni_data, E = loader.load_data("TODO PATH TO DATA", "PATH TO EMBEDDED", batch=(32, 200, 200, 32, 32))

		snli_train, snli_dev , snli_test = snli_data
		snli_train.repeat = False

		model = ResidualStackedEncoders(mlp_dim=self.mlp_dim, do_r=self.do_r)
		model.E.weight.data = E

		# TODO
		# if torch.cuda.is_available():
		# 	E.cuda()
		# 	model.cuda()

		start_lr = 2e-4
		optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=start_lr)
		loss_criteria = nn.CrossEntropyLoss()

		start = datetime.datetime.now()
		epochs = 0
		best_dev = -1


		## TODO SAVE PARAMS

		for i in range(EPOCHS):
			snli_train.init_epoch()
			train_it, dev_it = snli_train, snli_dev
			train_it.repeat = False
			# if i != 0 - > load data
			# start_performence = self.model_evaluate(model, dev_it, loss_criteria)
			lr = start_lr / (2 ** sqrt(i))
			for batch_i, batch in tqdm(enumerate(train_it)):
				model.train()
				s1, s1_l = batch.premise
				s2, s2_l = batch.hypothesis
				y = batch.label - 1

				output = model(s1, (s1_l-1), s2, (s2_l-1))
				loss = loss_criteria(output, y)
				optimizer.zero_grad()
				for param_group in optimizer.param_groups:
					param_group['lr'] = lr
				loss.backward()
				optimizer.step()
				mod = MOD_DICT[batch_i]
				if (i + batch_i) % mod == 0:
					model.word_size = 150
					dev_score, dev_loss = self.model_evaluate(model, dev_it, loss_criteria)
					if best_dev < dev_score:
						best_dev = dev_score
						# TODO PRINT results
			#path = os.path.join("ROOT", "filepath", "file name format")
			#torch.save(model.state_dict(), path)

	def model_evaluate(self, model, dev_it, loss_criteria):
		pass


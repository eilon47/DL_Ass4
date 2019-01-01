import random

from torch import nn
from torch.nn import Conv2d, MaxPool1d
import torch as tc
import torch.autograd as ag
from torch.utils.data import DataLoader, Subset


class CharLevelBiLSTM(nn.Module):
    def __init__(self, args):
        super(CharLevelBiLSTM, self).__init__()
        self.E = nn.Embedding(args["char_vocab_dim"], args["em_dim"])
        self.f1 = Conv2d(args["con_nn_in"], args["con_nn_out"], args["con_nn_kernel_1"], args["con_nn_stride"])
        self.f2 = Conv2d(args["con_nn_in"], args["con_nn_out"], args["con_nn_kernel_2"], args["con_nn_stride"])
        self.f3 = Conv2d(args["con_nn_in"], args["con_nn_out"], args["con_nn_kernel_3"], args["con_nn_stride"])
        self.mp1_fix = args["con_nn_kernel_1"][0] - 1
        self.mp2_fix = args["con_nn_kernel_2"][0] - 1
        self.mp3_fix = args["con_nn_kernel_3"][0] - 1

    def forward(self, x):
        mp1 = MaxPool1d(x.shape[2] - self.mp1_fix, 1)
        mp2 = MaxPool1d(x.shape[2] - self.mp2_fix, 1)
        mp3 = MaxPool1d(x.shape[2] - self.mp3_fix, 1)
        x = self.E(x).unsqueeze(dim=2)
        x1 = tc.stack([mp1(self.f1(x[i, :]).squeeze(dim=3)).squeeze(dim=1) for i in range(x.shape[0])])
        x2 = tc.stack([mp2(self.f2(x[i, :]).squeeze(dim=3)).squeeze(dim=1) for i in range(x.shape[0])])
        x3 = tc.stack([mp3(self.f3(x[i, :]).squeeze(dim=3)).squeeze(dim=1) for i in range(x.shape[0])])
        x = tc.cat([x1, x2, x3], dim=2)
        return x


def load_embedding(matrix, gpu=False, non_trainable=False):
    matrix = tc.Tensor(matrix).cuda() if gpu else tc.Tensor(matrix)
    num_e, e_dim = matrix.size()
    embedded_layer = nn.Embedding(num_e, e_dim)
    embedded_layer.load_state_dict({'weight': matrix})
    if non_trainable:
        embedded_layer.weight.requires_grad = False
    return embedded_layer


class SentenceLevelBiLSTM(nn.Module):
    def __init__(self, args):
        super(SentenceLevelBiLSTM, self).__init__()
        # word embed layer
        self.E = load_embedding(args["pre_trained_embedded"], args["gpu"]) if args["is_pre_trained"] \
            else nn.Embedding(args["word_vocab_dim"], args["em_dim"])
        # Bi-LSTM layer     TODO add skip connections
        self.bilstm = nn.LSTM(args["em_dim"]+args["em_char_dim"], args["lstm_dim"], args["lstm_layers"],
                                   dropout=args["dropout"], batch_first=True)

    def composition(self):
        w_hi, w_hf, w_hc, w_ho = self.bilstm.weight_hh_l0.chunk(4, 0)
        norm_output = tc.norm(w_hi, dim=1)
        attention_coefficient = norm_output / tc.sum(norm_output)
        return attention_coefficient

    def forward(self, words_embed, chr_rep):
        composition_out = self.composition()
        avg_pool = nn.AvgPool1d(words_embed.shape[1], 1)
        max_pool = nn.MaxPool1d(words_embed.shape[1], 1)

        x = self.E(words_embed)
        x = tc.cat([x, chr_rep], dim=2)
        output_seq, _ = self.bilstm(x)

        avg_pool = avg_pool(output_seq.transpose(1, 2)).squeeze(dim=2)
        max_pool = max_pool(output_seq.transpose(1, 2)).squeeze(dim=2)
        gate_attention = tc.sum(output_seq * composition_out, dim=1)
        x = tc.cat([gate_attention, avg_pool, max_pool], dim=1)
        return x


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        # useful info in forward function
        self.layer = nn.Linear(args["in_dim"], args["hid_dim"])
        self.out_layer = nn.Linear(args["hid_dim"], args["out_dim"])
        self.activation = args["sigmoid"]

    def forward(self, premise, hypothesis):
        x = tc.cat([premise, hypothesis, (premise - hypothesis), (premise * hypothesis)], dim=1)
        x = self.layer(x)
        x = self.activation(x)
        x = self.output_layer(x)
        x = nn.functional.softmax(x, dim=1)
        return x


class SNLIModel(nn.Module):
    def __init__(self, models_args):
        super(SNLIModel, self).__init__()
        self.char_level = CharLevelBiLSTM(models_args["CHAR_LEVEL"])
        self.premise_model = SentenceLevelBiLSTM(models_args["SEQ_LEVEL"])
        self.hypothesis_model = SentenceLevelBiLSTM(models_args["SEQ_LEVEL"])  # TODO change to hyp and prem
        self.mlp_model = MLP(models_args["MLP"])  # combine with (NN)
        self.optimizer = self.set_optimizer(models_args["LR"], models_args["OPTIMIZER"])

    # init optimizer with RMS_prop
    def set_optimizer(self, lr, opt):
        return opt(self.parameters(), lr=lr)

    def forward(self, premise_words, premise_char, hypothesis_words, hypothesis_char):
        premise_v = self.premise_model(premise_words, self.char_level(premise_char))
        hypothesis_v = self.hypothesis_model(hypothesis_words, self.char_level(hypothesis_char))
        return self.mlp_model(premise_v, hypothesis_v)


class SNLITrainer(object):
    def __init__(self, model, args, train_data=None, dev_data=None):
        self.model = model
        self.epochs = args["EPOCHS"]
        self.vrate = args["VRATE"]
        self.batch = args["BATCH"]
        self.gpu = args["GPU"]
        self.loss_func = args["LOSS_TYPE"]
        if self.gpu:
            self.model.cuda()
        self.load_data(train_data, dev_data)
        self.loss_dev = []
        self.loss_train = []
        self.acc_dev = []
        self.acc_train = []

    class Dummy:
        def __new__(self, d):
            return d

    def load_data(self, train_data, dev_data):
        self.train_loader, self.dev_loader = None, None
        if train_data is not None:
            self.train_loader = DataLoader(train_data, batch_size=self.batch,collate_fn=train_data.collate_fn,shuffle=True)
            self.train_valid = DataLoader(Subset(train_data, list(set(random.sample(range(1, len(train_data)), int(0.11 * len(train_data)))))),
                                          batch_size=self.batch, collate_fn=train_data.collate_fn)

        if dev_data is not None:
            self.dev_loader = DataLoader(dev_data, batch_size=self.batch, collate_fn=train_data.collate_fn, shuffle=True)

    def validate_train_and_dev(self, i):
        # validate Train
        loss, accuracy = self.validate(self.train_valid)
        self.loss_train.append((i, loss))
        self.acc_train.append((i, accuracy))
        # validate Dev
        if self.dev_loader is not None:
            loss, accuracy = self.validate(self.dev_loader)
            self.loss_dev.append((i, loss))
            self.acc_dev.append((i, accuracy))

    def train(self):
        if self.train_loader is None:
            print("Can't train due to None in train loader")
            return
        print("train...")
        self.loss_dev = []
        self.loss_train = []
        self.acc_dev = []
        self.acc_train = []
        for epoch in range(self.epochs):
            print("epoch: {}".format(epoch))
            # set model to train mode
            self.model.train()
            # calc number of iteration in current epoch
            len_data = len(self.train_loader)
            for i, (p, h, pw, hw, pc, hc, label) in enumerate(self.train_loader):
                print("\r\r\r%d" % int(100 * (i + 1) / len_data) + "%")
                pw, pc, hw, hc, label = self.create_input_to_model(pw, pc, hw, hc, label)
                self.model.zero_grad()
                output = self.model(pw, pc, hw, hc)
                loss = self.loss_func(output, label)
                loss.backward()
                self.model.optimizer.step()

                if self.vrate and i % self.vrate == 0:
                    print("validating dev in epoch:" + "\t" + str(epoch + 1) + "/" + str(self.epochs))
                    self.validate_train_and_dev(epoch + (i / len_data))

    def validate(self, data):
        print("validating...")
        loss, correct, total = 0, 0, 0
        self.model.eval()
        len_data = len(data)
        for i, (p, h, pw, hw, pc, hc, label) in enumerate(data):
            print("\r\r\r%d" % int(100 * (i + 1) / len_data) + "%")
            pw, pc, hw, hc, label = self.create_input_to_model(pw, pc, hw, hc, label)
            output = self.model(pw, pc, hw, hc)
            loss += self.loss_func(output, label)
            # calculate accuracy
            correct += sum([1 if out.item() == lab.item() else 0 for out, lab in zip(tc.argmax(output, dim=1), label)])
            total += label.shape[0]

        loss = float(loss / len(data))
        accuracy = correct / total
        print("loss is {}, accuracy is {}".format(loss,accuracy))
        return loss, accuracy

    def create_input_to_model(self, pwords, pchars, hwords, hchars, label, is_test=False):
        if is_test:
            cls = self.Dummy
        else:
            cls = ag.Variable
        pwords = cls(pwords).cuda() if self.gpu else cls(pwords)
        pchars = cls(pchars).cuda() if self.gpu else cls(pchars)
        hwords = cls(hwords).cuda() if self.gpu else cls(hwords)
        hchars = cls(hchars).cuda() if self.gpu else cls(hchars)
        label = cls(label).cuda() if self.gpu else cls(label)
        return pwords, pchars, hwords, hchars, label

    def predict(self, dataset):
        self.model.eval()
        test_data = DataLoader(dataset,batch_size=self.batch,collate_fn=dataset.collate_fn,shuffle=False)
        with tc.no_grad():
            loss, correct, total = 0, 0, 0
            results = []
            len_data = len(test_data)
            for i, (p, h, pw, hw, pc, hc, label) in enumerate(test_data):
                print("\r\r\r%d" % int(100 * (i + 1) / len_data) + "%")
                pw, pc, hw, hc, label = self.create_input_to_model(pw, pc, hw, hc, label, is_test=True)
                output = self.model(pw, pc, hw, hc)
                loss += self.loss_func(output, label)
                correct += sum([1 if out.item() == lab.item() else 0 for out, lab in zip(tc.argmax(output, dim=1), label)])
                total += label.shape[0]
                for premise, hypothesis, pred in zip(p, h, tc.argmax(output, dim=1)):
                    results.append((dataset.label(pred.item()), " ".join(premise), " ".join(hypothesis)))
            loss = float(loss / len(test_data))
            accuracy = correct / total
            print("loss is {}, accuracy is {}".format(loss, accuracy))
        return results, loss, accuracy



















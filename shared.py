import os
import torch

compress = "zip"
data_dir = os.path._getfullpathname(__file__)
data_dir = os.path.split(data_dir)[0]
data_dir = os.path.join(data_dir, "data")
glove = "glove.6B.50d"
glove_6b = glove+ ".{}"
glove_dir = os.path.join(data_dir, "GloVe_vocab")
glove_compressed = os.path.join(glove_dir, glove_6b.format(compress))
glove_txt = os.path.join(glove_dir, glove_6b.format("txt"))
glove_download_url = ""

snli_download_url = ""
snli_dir1 =  os.path.join(data_dir, "snli_1.0")
snli_dir = os.path.join(data_dir, "snli_1.0", "snli_1.0")
snli_compressed = os.path.join(snli_dir, "snli_1.0.{}".format(compress))
snli_file = os.path.join(snli_dir, "snli_1.0_{}.txt")
snli_train = snli_file.format("train")
snli_dev = snli_file.format("dev")
snli_test = snli_file.format("test")

LABELS = {
"ENTAILMENT" :"entailment",
"CONTRADICTION" :"contradiction",
"NATURAL" : "neutral"
}
START = "$start$"
END = "$end$"
UNK = "UUUNKKK"
PAD = "$pad$"
GPU = False
EM_DIM = 50
NUM_LAYERS = 3
CHAR_VOCAB_DIM = 129
LR = 0.01
IN_DIM = 600
HID_DIM = 300
OPTIMIZER = torch.optim.Adam

# MODELS = {
#     "MLP":{ "in_dim":IN_DIM, "out_dim":len(LABELS), "hid_dim":HID_DIM, "sigmoid":torch.tanh },
#     "CHAR_LEVEL":{"em_dim":30, "char_vocab_dim":CHAR_VOCAB_DIM, "con_nn_in":1, "con_nn_out":1, "con_nn_stride":1,
#                   "con_nn_kernel_1":(3, 30),"con_nn_kernel_2":(4, 30),"con_nn_kernel_3":(5, 30)},
#     "SEQ_LEVEL":{"pre_trained_embedded":vocab_file, "gpu":False, "is_pre_trained":True, "em_dim":EM_DIM, "em_char_dim":3,
#                  "word_vocab_dim":129, "lstm_dim":EM_DIM, "lstm_layers":NUM_LAYERS, "dropout":0.3},
#     "LR":LR,
#     "OPTIMIZER": torch.optim.Adam
# }
#
#
# SNLI_PARAMS = {
#     "LOSS_TYPE":torch.nn.CrossEntropyLoss,
#     "GPU": False,
#     "BATCH": 256,
#     "EPOCHS": 20,
#     "VRATE":200
# }


FORMAT = "{} parameters are: "

class CnnCharArgs(object):
    def __init__(self, vocab_dim=129):
        self.em_dim = 30
        self.char_vocab_dim = vocab_dim
        self.con_nn_in, self.con_nn_out, self.con_nn_stride =1,1,1
        self.con_nn_kernel_1, self.con_nn_kernel_2, self.con_nn_kernel_3 = (3, self.em_dim), (4, self.em_dim),\
                                                                           (5, self.em_dim)
    def __str__(self):
        s = FORMAT.format(self.__class__.__name__)
        s += "EM DIM = {}, CHAR VOCAB DIM = {}, CONN NN = ({}, {}), STRIDE = {}, KERNELS = ({},{},{})".format(
            self.em_dim, self.char_vocab_dim, self.con_nn_in,self.con_nn_out,
            self.con_nn_stride, self.con_nn_kernel_1, self.con_nn_kernel_2, self.con_nn_kernel_3)
        return s


class SeqArgs(object):
    def __init__(self, word_vocab_dim=129, pre_trained_embedded=None, gpu=GPU):
        self.pre_trained_embedded = pre_trained_embedded
        self.is_pre_trained = True if pre_trained_embedded is not None else False
        self.lstm_dim = 50
        self.em_dim = 50
        self.em_char_dim = 3
        self.word_vocab_dim = word_vocab_dim
        self.lstm_layers = 3
        self.dropout = 0.3
        self.gpu = gpu

    def __str__(self):
        s = FORMAT.format(self.__class__.__name__)
        s += "EM DIM = {}, WORD VOCAB DIM = {}, PRE TRAINED = {}, EM CHAR DIM = {}, LSTM DIM = {}, DROPOUT =" \
             " {}, GPU = {})".format(self.em_dim, self.word_vocab_dim, self.pre_trained_embedded,
                                     self.em_char_dim,self.lstm_dim, self.dropout, self.gpu)
        return s


class MlpArgs(object):
    def __init__(self):
        self.in_dim = 1200
        self.hid_dim = 300
        self.out_dim = 3
        self.activation = torch.tanh

    def __str__(self):
        s = FORMAT.format(self.__class__.__name__)
        s += "IN DIM = {}, HID DIM = {}, OUT DIM = {})".format(self.in_dim, self.hid_dim, self.out_dim)
        return s


class SnliArgs(object):
    def __init__(self, char_level_params, premise_params, hypo_params, mlp_params, lr=LR, optim=OPTIMIZER):
        self.char_level_params = char_level_params
        self.premise_params = premise_params
        self.hypo_params = hypo_params
        self.mlp_params = mlp_params
        self.LEARNING_RATE = lr
        self.OPTIMIZER = optim

    def __str__(self):
        s = FORMAT.format(self.__class__.__name__)
        s += "\n\t\t\t" + self.char_level_params
        s += "\n\t\t\t" + self.premise_params
        s += "\n\t\t\t" + self.hypo_params
        s += "\n\t\t\t" + self.mlp_params
        s += "\n" + "LR = {}, OPTIMIZER = {}".format(self.LEARNING_RATE, self.OPTIMIZER.__class__.__name__)
        return s


class TrainerArgs(object):
    def __init__(self, gpu=GPU):
        self.loss = torch.nn.functional.cross_entropy
        self.batch = 256
        self.gpu = gpu
        self.epochs = 20
        self.vrate = 200

    def __str__(self):
        s = FORMAT.format(self.__class__.__name__)
        s += "LOSS = {}, BATCH = {}, GPU = {}, EPOCHS = {}, VRATE = {}".format(self.loss.__class__.__name__,
                                                                        self.batch, self.gpu, self.epochs, self.vrate)
        return s






















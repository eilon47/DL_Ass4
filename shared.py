import os
import torch

data_dir = os.path._getfullpathname(__file__)
data_dir = os.path.split(data_dir)[0]
data_dir = os.path.join(data_dir, "data")
glove_6b = "glove.6B.50d.txt"
globe_dir = "GloVe_vocab"
snli_file = os.path.join(data_dir, "snli_1.0", "snli_1.0_{}.txt")

# DATA FILES
vocab_file = os.path.join(data_dir, globe_dir, glove_6b)
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

EM_DIM = 50
NUM_LAYERS = 3
CHAR_VOCAB_DIM = 129
LR = 0.01
IN_DIM = 600
HID_DIM = 300

MODELS = {
    "MLP":{ "in_dim":IN_DIM, "out_dim":len(LABELS), "hid_dim":HID_DIM, "sigmoid":torch.tanh },
    "CHAR_LEVEL":{"em_dim":30, "char_vocab_dim":CHAR_VOCAB_DIM, "con_nn_in":1, "con_nn_out":1, "con_nn_stride":1,
                  "con_nn_kernel_1":(3, 30),"con_nn_kernel_2":(4, 30),"con_nn_kernel_3":(5, 30)},
    "SEQ_LEVEL":{"pre_trained_embedded":None, "gpu":True, "is_pre_trained":False, "em_dim":EM_DIM, "em_char_dim":3,
                 "word_vocab_dim":129, "lstm_dim":EM_DIM, "lstm_layers":NUM_LAYERS, "dropout":0.3},
    "LR":LR,
    "OPTIMIZER": torch.optim.Adam
}


SNLI_PARAMS = {
    "LOSS_TYPE":torch.nn.CrossEntropyLoss,
    "GPU": True,
    "BATCH": 256,
    "EPOCH": 20,
    "VRATE":200
}


























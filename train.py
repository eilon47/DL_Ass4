import os
import pickle
import torch as tc
from models import SNLITrainer, SNLIModel
from data_handler import SNLI
import shared
from shared import SeqArgs, CnnCharArgs, MlpArgs, TrainerArgs, SnliArgs



def load_datasets(train_src, pre_trained_src, dev_src=None):
    dev = None
    train = SNLI(train_src, pre_trained_src)
    if dev_src:
        dev = SNLI(dev_src)
        dev.load_word_vocabulary((train.word_embedded, train.index2word))
    return train, dev


def model(dataset:SNLI):
    params = SnliArgs(char_level_params=CnnCharArgs(vocab=dataset.char_len),
                      seq_params1=SeqArgs(word_vocab_dim=len(dataset.index2word), pre_trained_embedded=dataset.E),
                      seq_params2=SeqArgs(word_vocab_dim=len(dataset.index2word)),
                      mlp_params=MlpArgs())
    return SNLIModel(params)


def get_trainer(m, train, dev):
    params = TrainerArgs()
    return SNLITrainer(model=m,args=params,train_data=train, dev_data=dev)


def get_new_model_trainer(train_src, pre_trained_src, dev_src=None):
    train, dev = load_datasets(train_src, pre_trained_src, dev_src)
    mod = model(train)
    print("trainer is ready")
    return get_trainer(mod, train, dev)


def test_dataset(file, vocab):
    test = SNLI(file)
    test.load_word_vocabulary(vocab)
    return test


def write_to_file(model_name, results, loss, accuracy):
    loss_acc = open(model_name + "_loss_acc.txt", "wt")
    loss_acc.write("accuracy = " + str(accuracy) + "\nloss = " + str(loss))
    loss_acc.close()


def load_trainer_and_args(file):
    model, params, vocab = pickle.load(open(os.path.join("..", "pkl", file + ".trained_model"), "rb"))
    return SNLITrainer(model, params), params, vocab


def get_trained_model_trainer(model_file, test_file):
    trainer, params, vocab = load_trainer_and_args(model_file)
    test = test_dataset(test_file, vocab)
    return trainer, test


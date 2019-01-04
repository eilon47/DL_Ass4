import os
import pickle
import torch as tc
from models import SNLITrainer, SNLIModel
from data_handler import SNLI
import shared
from shared import SeqArgs, CnnCharArgs, MlpArgs, TrainerArgs, SnliArgs


def get_trainer(train, dev, pre_trained_file):
    train_ds = SNLI(train, pre_trained_file)
    dev_ds = SNLI(dev, pre_trained_file)

    charparams = CnnCharArgs(vocab_dim=train_ds.char_len)
    premisparams, hypparams = SeqArgs(word_vocab_dim=train_ds.data_legth, pre_trained_embedded=train_ds.E), SeqArgs(
        word_vocab_dim=train_ds.data_legth, pre_trained_embedded=train_ds.E)
    mlpparams = MlpArgs()
    print(mlpparams)
    model = SNLIModel(SnliArgs(charparams, premisparams, hypparams, mlpparams))

    snli_trainer = SNLITrainer(model, TrainerArgs(),train_ds, dev_ds)
    return snli_trainer


def get_trainer_with_data(model_file):
    model, params, vocab = pickle.load(open(os.path.join("..", "pkl", model_file + ".trained_model"), "rb"))
    return SNLITrainer(model, params), params, vocab


def test_ds(test_src, vocab):
    test = SNLI(test_src)
    test.load(vocab)


def write_results_to_file(file, results, loss, acc):
    fd = open(file+"loss_and_acc", "w")
    fd.write("accuracy is {}, loss is {}".format(loss, acc))
    fd.close()
    fd = open(file+"_predictions", "w")
    for l,p,h in results:
        fd.write("{}\t\t{}\t\t{}\n".format(l,p,h))
    fd.close()

if __name__ == '__main__':
    print(shared.snli_dev)
    print(shared.snli_train)
    train = open(shared.snli_train, 'r')
    dev = open(shared.snli_dev, 'r')
    trainer = get_trainer(train, dev, shared.vocab_file)
    print("in train")
    trainer.train()


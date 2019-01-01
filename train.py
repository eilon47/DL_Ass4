import os
import pickle
import torch as tc
from models import SNLITrainer, SNLIModel
from data_handler import SNLI
from shared import MODELS, SNLI_PARAMS


def get_trainer(train, dev, is_pre_trained=False):
    snli_model = SNLIModel(MODELS)
    train_ds = SNLI(train, is_pre_trained)
    dev_ds = SNLI(dev)
    dev_ds.load(dev)
    snli_trainer = SNLITrainer(snli_model, SNLI_PARAMS,train_ds, dev_ds)
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
    print(tc.cuda.is_available())




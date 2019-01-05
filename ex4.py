import os
import zipfile
import requests

from data_handler import SNLI
from shared import snli_train, snli_dev, snli_test, glove_txt
import shared
import train

def create_directories():
    print("Checking all directories exists")
    f = "{} is not exists, creating it"
    if not os.path.exists(shared.data_dir):
        os.mkdir(shared.data_dir)
        print(f.format(shared.data_dir))
    if not os.path.exists(shared.glove_dir):
        os.mkdir(shared.glove_dir)
        print(f.format(shared.glove_dir))
    if not os.path.exists(shared.snli_dir):
        os.mkdir(shared.snli_dir)
        print(f.format(shared.snli_dir))


def get_data_files():
    if not os.path.exists(shared.snli_compressed):
        snli = [snli_test, snli_dev, snli_train]
        snli = [os.path.exists(s) for s in snli]
        if not all(snli):
            print("One of the snli files is not exists - Download starts")
            path = os.path.curdir
            os.chdir(shared.snli_dir)
            requests.get("https://nlp.stanford.edu/projects/snli/snli_1.0.zip")
            if not os.path.exists(shared.snli_compressed):
                print("Problem in downloading file https://nlp.stanford.edu/projects/snli/snli_1.0.zip, exiting with error")
                exit(-1)
            zip_file = zipfile.ZipFile(shared.snli_compressed)
            print("Extracting snli files")
            zip_file.extractall(shared.os.curdir)
            os.chdir(path)
    if not os.path.exists(shared.glove_compressed):
        if not os.path.exists(shared.glove_txt):
            path = os.path.curdir
            os.chdir(shared.glove_dir)
            requests.get("http://nlp.stanford.edu/data/wordvecs/"+shared.glove_6b.format("zip"))
            if not os.path.exists(shared.snli_compressed):
                print("Problem in downloading file " +
                      "http://nlp.stanford.edu/data/wordvecs/"+shared.glove_6b.format("zip") + ", exiting with error")
                exit(-1)
            zip_file = zipfile.ZipFile(shared.snli_compressed)
            print("Extracting Glove files")
            zip_file.extractall(shared.os.curdir)
            os.chdir(path)


def prepare():
    print("Preparing environment")
    create_directories()
    get_data_files()
    print("Done preparing environment successfully")



def write_results_to_file(file, results, loss, acc):
    fd = open(file+"-info", "w")
    fd.write("accuracy is {}, loss is {}".format(loss, acc))
    fd.close()
    fd = open(file+"-pred", "w")
    for l,p,h in results:
        fd.write("{}\t\t{}\t\t{}\n".format(l,p,h))
    fd.close()

def main(args):
    prepare()
    trainer = train.get_trainer(snli_train, snli_dev, glove_txt)
    trainer.train()
    results, loss, accuracy = trainer.predict(SNLI(open(snli_test, glove_txt)))
    write_results_to_file("first_time", results, loss, accuracy)
    print("Done!!!!!!")


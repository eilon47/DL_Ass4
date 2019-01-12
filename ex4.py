import os
import sys
import time
import zipfile
import requests
from torch.nn import LSTM

from data_handler import SNLI
from shared import snli_train, snli_dev, snli_test, glove_txt
import shared
import train
import optparse

def create_directories():
    print("Checking all directories exists")
    f = "{} is not exists, creating it"
    if not os.path.exists(shared.data_dir):
        os.mkdir(shared.data_dir)
        print(f.format(shared.data_dir))
    if not os.path.exists(shared.glove_dir):
        os.mkdir(shared.glove_dir)
        print(f.format(shared.glove_dir))
    if not os.path.exists(shared.snli_dir1):
        os.mkdir(shared.snli_dir1)
        print(f.format(shared.snli_dir1))


def download_file(url, file_name):
    with open(file_name, "wb") as f:
        print("Downloading "+ file_name)
        response = requests.get(url, stream=True)
        total_length = response.headers.get('content-length')

        if total_length is None:  # no content length header
            f.write(response.content)
        else:
            dl = 0
            total_length = int(total_length)
            for data in response.iter_content(chunk_size=4096):
                dl += len(data)
                f.write(data)
                done = int(100 * dl / total_length)
                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (100 - done)))
                sys.stdout.write(str(done) + "%")
                sys.stdout.flush()
        print("")
    if not os.path.exists(file_name):
        print("Problem in downloading file {}, exiting with error".format(url))
        exit(-1)


def extract_file(file_path, folder, members=None):
    zip_file = zipfile.ZipFile(file_path, 'r')
    try:
        print("Extracting file: " + file_path)
        zip_file.extractall(folder, members=members)
    except Exception as e:
        print(e)
        print("Problem in extracting file " + file_path + " to directory " + folder + ", exiting with error")
        exit(-1)


def get_data_files():
    snli = [snli_test, snli_dev, snli_train]
    snli = [os.path.exists(s) for s in snli]
    path = os.path.curdir
    if not all(snli):
        print("One of the snli files is not exists")
        if not os.path.exists(shared.snli_compressed):
            print(shared.snli_compressed + " is not found, downloading")
            os.chdir(shared.snli_dir1)
            download_file("https://nlp.stanford.edu/projects/snli/snli_1.0.zip","snli_1.0.zip")
        snli = [snli_test, snli_dev, snli_train]
        snli = ["snli_1.0/" + os.path.split(s)[1] for s in snli]
        extract_file(shared.snli_compressed, shared.snli_dir1, snli)
        os.chdir(path)
    path = os.path.curdir
    if not os.path.exists(shared.glove_txt):
        if not os.path.exists(shared.glove_compressed):
            os.chdir(shared.glove_dir)
            download_file("http://nlp.stanford.edu/data/glove.6B.zip", "glove.6B.zip")
            if not os.path.exists(shared.snli_compressed):
                print("Problem in downloading file " +
                      "http://nlp.stanford.edu/data/glove.6B.zip", "glove.6B.zip" + ", exiting with error")
                exit(-1)
        members = [shared.glove_6b.format("txt")]
        extract_file(shared.glove_compressed, shared.glove_dir, members=members)
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

current_time = str(time.strftime("%H_%M_%S", time.gmtime(time.time())))

option_parser = optparse.OptionParser()
option_parser.add_option("--train", dest="train", help="training new model, possible to add name for the model ",
                         default="model_{}".format(current_time))
option_parser.add_option("--test", dest="test", help="test run, possible to add model path to load (.pkl)", default=None)
option_parser.add_option("--run", dest="run", help="running training and test after that", default=False, action="store_true")


def main():
    options, args = option_parser.parse_args()
    prepare()
    if options.run:
        trainer, params = train.get_new_model_trainer(glove_txt, open(snli_train, "r"),open(snli_dev, "r"))
        trainer.train()
        train.save(trainer, options.train, params, trainer.train_data)
        results, loss, accuracy = trainer.predict(SNLI(open(snli_test, "r"), glove_txt))
        write_results_to_file("results", results, loss, accuracy)
    else:
        if options.train:
            trainer, params = train.get_new_model_trainer(glove_txt, open(snli_train, "r"), open(snli_dev, "r"))
            trainer.train()
            train.save(trainer, options.train, params, trainer.train_data)
        if options.test:
            trainer, params, vocab = train.load_trainer_and_args(options.test)
            results, loss, accuracy = trainer.predict(SNLI(open(snli_test, "r"), glove_txt))
            write_results_to_file("results", results, loss, accuracy)

if __name__ == '__main__':
    main()


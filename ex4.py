import os
import sys
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
                done = int(50 * dl / total_length)
                sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                sys.stdout.flush()


def get_data_files():
    if not os.path.exists(shared.snli_compressed):
        snli = [snli_test, snli_dev, snli_train]
        snli = [os.path.exists(s) for s in snli]
        if not all(snli):
            print("One of the snli files is not exists - Download starts")
            path = os.path.curdir
            os.chdir(shared.snli_dir)
            download_file("https://nlp.stanford.edu/projects/snli/snli_1.0.zip","snli_1.0.zip")
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
            download_file("http://nlp.stanford.edu/data/wordvecs/"+shared.glove_6b.format("zip"),shared.glove_6b.format("zip"))
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

def main():
    prepare()
    trainer = train.get_trainer(snli_train, snli_dev, glove_txt)
    trainer.train()
    results, loss, accuracy = trainer.predict(SNLI(open(snli_test, glove_txt)))
    write_results_to_file("first_time", results, loss, accuracy)
    print("Done!!!!!!")

if __name__ == '__main__':
    main()
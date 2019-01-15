###################################################################################################
This app is an implementation of the paper:

Recurrent Neural Network-Based Sentence Encoder with Gated Attention for Natural Language Inference
by Qian Chen, Xiaodan Zhu, Zhen-Hua Ling, Si Wei, Hui Jiang, Diana Inkpen.

This paper can be found here : https://arxiv.org/pdf/1708.01353.pdf

###################################################################################################


###################### Authors #######################
Eilon Bashari 308576933     Daniel Greenspan 308243948
######################################################


To run this app please use the flags we created as it explain in the help menu (to run the help please run python ex4.py --help).
Options:
  -h, --help     show this help message and exit
  --train=TRAIN  training new model, possible to add name for the model
  --test=TEST    test run, possible to add model name to load (.pkl)
  --run          running training and test after that

Please make sure the directory has the structure as explained next. Please notice that if one of the required files is
missing, this app will download the compressed file automatically.
If you like to avoid it you can download it by yourself from :
-https://nlp.stanford.edu/projects/snli/snli_1.0.zip
-http://nlp.stanford.edu/data/glove.6B.zip
and put them in the right directories.
If you didn't extract the files inside the app will do it by itself.


We recommend to run the command:
    python ex4.py --run


Folder structure
1.data:
    GloVe:
        -glove.6B.zip - optional
        -globe.6B.300d.txt  - required
    snli_1.0:
        snli_1.0: - required
            snli_1.0_dev.txt
            snli_1.0_train.txt
            snli_1.0_test.txt
        snli_1.0.zip - optional
2.python source files - required
3.README.txt



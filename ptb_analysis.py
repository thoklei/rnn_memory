import tensorflow as tf
import numpy as np
def _count_words_per_sentence(filename):
    with tf.gfile.GFile(filename, "r") as f:
            return [len(l.split()) for l in f.read().replace("\n", "<eos>").split("<eos>")]

def main(_):
    res = _count_words_per_sentence("/Users/thomasklein/Uni/Bachelorarbeit/experiments/data/simple-examples/data/ptb.train.txt")
    print(res)
    resarray = np.asarray(res)
    mean = np.mean(resarray)
    std = np.std(resarray)
    print("The mean sentence length is {}, with a standard deviation of {}.".format(mean, std))

def count():
    with tf.gfile.GFile("/Users/thomasklein/Uni/Bachelorarbeit/experiments/data/simple-examples/data/ptb.test.txt", "r") as f:
        print(len(f.read().replace("\n", "<eos>").split()))

count()
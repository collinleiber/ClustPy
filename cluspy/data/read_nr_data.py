import numpy as np
import os

def _laod_nr_data(file_name, n_labels):
    path = os.path.dirname(__file__) + "/datasets/" + file_name
    dataset = np.genfromtxt(path, delimiter=",")
    data = dataset[:,n_labels:]
    labels = dataset[:,:n_labels]
    return data, labels

def load_aloi():
    return _laod_nr_data("aloi.data", 2)

def load_fruit():
    return _laod_nr_data("fruit.data", 2)

def load_nrletters():
    return _laod_nr_data("nrLetters.data", 3)

def load_stickfigures():
    return _laod_nr_data("stickfigures.data", 2)


import numpy as np
import os
from cluspy.data.real_world_data import _get_download_dir, _download_file
import tarfile
import re
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.feature_selection import VarianceThreshold
from nltk.stem import SnowballStemmer
from PIL import Image


def _laod_nr_data(file_name, n_labels):
    path = os.path.dirname(__file__) + "/datasets/" + file_name
    dataset = np.genfromtxt(path, delimiter=",")
    data = dataset[:, n_labels:]
    labels = dataset[:, :n_labels]
    return data, labels


def load_aloi():
    return _laod_nr_data("aloi.data", 2)


def load_fruit():
    return _laod_nr_data("fruit.data", 2)


def load_nrletters():
    return _laod_nr_data("nrLetters.data", 3)


def load_stickfigures():
    return _laod_nr_data("stickfigures.data", 2)


def load_cmu_faces(downloads_path=None):
    directory = _get_download_dir(downloads_path) + "/cmufaces/"
    filename = directory + "faces_4.tar.gz"
    if not os.path.isfile(filename):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        _download_file("http://archive.ics.uci.edu/ml/machine-learning-databases/faces-mld/faces_4.tar.gz",
                       filename)
        # Unpack zipfile
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall(directory)
    names = np.array(
        ["an2i", "at33", "boland", "bpm", "ch4f", "cheyer", "choon", "danieln", "glickman", "karyadi", "kawamura",
         "kk49", "megak", "mitchell", "night", "phoebe", "saavik", "steffi", "sz24", "tammo"])
    positions = np.array(["straight", "left", "right", "up"])
    expressions = np.array(["neutral", "happy", "sad", "angry"])
    eyes = np.array(["open", "sunglasses"])
    data_list = []
    label_list = []
    for name in names:
        path_images = directory + "/faces_4/" + name
        for image in os.listdir(path_images):
            if not image.endswith("_4.pgm"):
                continue
            # get image data
            image_data = Image.open(path_images + "/" + image)
            image_data_vector = np.array(image_data).reshape(image_data.size[0] * image_data.size[1])
            # Get labels
            name_parts = image.split("_")
            user_id = np.argwhere(names == name_parts[0])[0][0]
            position = np.argwhere(positions == name_parts[1])[0][0]
            expression = np.argwhere(expressions == name_parts[2])[0][0]
            eye = np.argwhere(eyes == name_parts[3])[0][0]
            label_data = np.array([user_id, position, expression, eye])
            # Save data and labels
            data_list.append(image_data_vector)
            label_list.append(label_data)
    labels = np.array(label_list)
    data = np.array(data_list)
    return data, labels


"""
Load WebKB
"""


def load_webkb(remove_headers=True, use_categories=["course", "faculty", "project", "student"],
               use_universities=["cornell", "texas", "washington", "wisconsin"], downloads_path=None):
    directory = _get_download_dir(downloads_path) + "/WebKB/"
    filename = directory + "webkb-data.gtar.gz"
    if not os.path.isfile(filename):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        _download_file("http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/webkb-data.gtar.gz",
                       filename)
        # Unpack zipfile
        with tarfile.open(filename, "r:gz") as tar:
            for obj in tar.getmembers():
                if obj.isdir():
                    # Create Directory
                    tar.extract(obj, directory)
                else:
                    # Can not handle filenames with special characters. Therefore, rename files
                    new_name = obj.name.replace("~", "_").replace(".", "_").replace("^", "_").replace(":", "_").replace(
                        "\r", "")
                    # Get file content
                    f = tar.extractfile(obj)
                    lines = f.readlines()
                    # Write file
                    with open(directory + new_name, "wb") as output:
                        for line in lines:
                            output.write(line)
    texts = []
    labels = np.empty((0, 2), dtype=np.int)
    hmtl_tags = re.compile(r'<[^>]+>')
    head_tags = re.compile(r'MIME-Version:[:,./\-\w\s]+<html>')
    number_tags = re.compile(r'\d*')
    # Read files
    for i, category in enumerate(use_categories):
        for j, univerity in enumerate(use_universities):
            inner_directory = "{0}webkb/{1}/{2}/".format(directory, category, univerity)
            files = os.listdir(inner_directory)
            for file in files:
                with open(inner_directory + file, "r") as f:
                    lines = f.read()
                    if remove_headers:
                        # Remove header
                        lines = head_tags.sub('', lines)
                    # Remove HTML tags
                    lines = hmtl_tags.sub('', lines)
                    lines = number_tags.sub('', lines)
                    texts.append(lines)
                    labels = np.r_[labels, [[i, j]]]
    # Execute TF-IDF and remove stop-words
    vectorizer = _StemmedCountVectorizer(dtype=np.float64, stop_words="english", min_df=0.01)
    data_sparse = vectorizer.fit_transform(texts)
    selector = VarianceThreshold(0.25)  # 0.25 ohne min_df
    data_sparse = selector.fit_transform(data_sparse)
    tfidf = TfidfTransformer(sublinear_tf=True)
    data_sparse = tfidf.fit_transform(data_sparse)
    data = np.asarray(data_sparse.todense())
    return data, labels


class _StemmedCountVectorizer(CountVectorizer):

    def build_analyzer(self):
        stemmer = SnowballStemmer('english')
        analyzer = super(_StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: (stemmer.stem(word) for word in analyzer(doc))

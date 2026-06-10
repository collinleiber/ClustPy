try:
    import requests
except:
    print("[WARNING] Could not import requests in clustpy.data._utils. Please install requests by 'pip install requests' if necessary")
try:
    from nltk.stem import SnowballStemmer
except:
    print(
        "[WARNING] Could not import nltk in clustpy.data.real_world_data to use the SnowballStemmer. Please install nltk by 'pip install nltk' if necessary")
try:
    from PIL import Image
except:
    print(
        "[WARNING] Could not import PIL in clustpy.data.real_world_data. Please install PIL by 'pip install Pillow' if necessary")
import numpy as np
import os
from pathlib import Path, PurePath
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import fetch_file
import subprocess


DEFAULT_DOWNLOAD_PATH = Path.home() / "Downloads" / "clustpy_datafiles"


def _get_download_dir(downloads_path: str | Path) -> Path:
    """
    Helper function to define the path where the data files should be stored. If downloads_path is None then default path
    '[USER]/Downloads/clustpy_datafiles' will be used. If the directory does not exists it will be created.

    Parameters
    ----------
    downloads_path : str | Path
        path to the directory where the data will be stored. Can be None

    Returns
    -------
    downloads_path : str
        path to the directory where the data will be stored. If input was None this will be equal to
        '[USER]/Downloads/clustpy_datafiles'
    """
    if downloads_path is None:
        env_data_path = os.environ.get("CLUSTPY_DATA", None)
        if env_data_path is None:
            downloads_path = DEFAULT_DOWNLOAD_PATH
        else:
            downloads_path = Path(env_data_path)
    elif isinstance(downloads_path, str):
        # Cast str to Path
        downloads_path = Path(downloads_path)
    if not downloads_path.is_dir():
        downloads_path.mkdir(parents=True, exist_ok=False)
        with open(downloads_path / "info.txt", "w") as f:
            f.write("This directory was created by the ClustPy python package to store real world data sets.\n"
                    "The default directory is '[USER]/Downloads/clustpy_datafiles' and can be changed with the "
                    "'downloads_path' parameter when loading a data set.\n"
                    "Alternatively, a global python environment variable for the path can be defined with os.environ['CLUSTPY_DATA'] = 'PATH'.")
    return downloads_path


def _download_file(file_url: str, filename_local: str | Path) -> None:
    """
    Helper function to download a file into a specified location.

    Parameters
    ----------
    file_url : str
        URL of the file
    filename_local : str | Path
        local name of the file after it has been downloaded
    """
    if isinstance(filename_local, str):
        filename_local = Path(filename_local)
    local_dir = filename_local.parent
    local_filename = filename_local.name
    print("Downloading data set from {0} to {1}".format(file_url, filename_local))
    fetch_file(file_url, folder=local_dir, local_filename=local_filename)


def _download_file_from_google_drive(file_id: str, filename_local: str | Path, chunk_size: int = 32768) -> None:
    """
    Download a file from google drive.
    Code taken from:
    https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url

    Parameters
    ----------
    file_id : str
        ID of the file on google drive
    filename_local : str | Path
        local name of the file after it has been downloaded
    chunk_size : int
        chink size when downloading the file (default: 32768)
    """
    print("Downloading data set {0} from Google Drive to {1}".format(file_id, filename_local))
    URL = "https://drive.google.com/uc"
    session = requests.Session()
    response = session.get(URL, params={"id": file_id, "confirm": "t"}, stream=True)
    if response.text.startswith("<!DOCTYPE"):
        # Large files can not be obtained automatically but need a second request
        try:
            URL_extracted = response.text.split("download-form\" action=\"")[1].split("\" method=\"get\"")[0]
            uuid = response.text.split("name=\"uuid\" value=\"")[1].split("\">")[0]
        except:
            raise Exception("[ERROR] New URL and UUID could not be extracted from first request in _download_file_from_google_drive")
        response = session.get(URL_extracted, params={"id": file_id, "confirm": "t", "uuid": uuid}, stream=True)
    with open(filename_local, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
    session.close()


def _load_data_file(filename_local: Path, file_url: str, delimiter: str = ",", last_column_are_labels: bool = True) -> tuple[
        np.ndarray, np.ndarray]:
    """
    Helper function to load a data file. Either the first or last column, depending on last_column_are_labels, of the
    data file is used as the label column.
    If file does not exist on the local machine it will be downloaded.

    Parameters
    ----------
    filename_local : Path
        local name of the file after it has been downloaded
    file_url : str
        URL of the file
    delimiter : str
        delimiter in the data file (default: ";")
    last_column_are_labels : bool
        specifies if the last column contains the labels. If false labels should be contained in the first column (default: True)

    Returns
    -------
    data, labels : tuple[np.ndarray, np.ndarray]
        the data numpy array, the labels numpy array
    """
    if not filename_local.is_file():
        _download_file(file_url, filename_local)
    datafile = np.genfromtxt(filename_local, delimiter=delimiter)
    if last_column_are_labels:
        data = datafile[:, :-1]
        labels = datafile[:, -1]
    else:
        data = datafile[:, 1:]
        labels = datafile[:, 0]
    # Convert labels to int32 format
    labels = labels.astype(np.int32)
    return data, labels


def _decompress_z_file(filename: str | Path, directory: str | Path) -> bool:
    """
    Helper function to decompress a 7z file. The function uses an installed version of 7zip to decompress the file.
    If 7zip is not installed on this machine, the function will return False and a warning is printed.

    Parameters
    ----------
    filename : str
        name of the file that should be decompressed
    directory : str
        directory of the file that should be decompressed

    Returns
    -------
    successful : bool
        True if decompression was successful, else False
    """
    if isinstance(filename, str):
        filename = Path(filename)
    if isinstance(directory, str):
        directory = Path(directory)
    cmd = ["7z", "x", filename.as_posix(), f"-o{directory.as_posix()}"]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[WARNING] 7Zip extraction failed or 7z executable is missing!")
        return False
    if not filename.with_suffix('').is_file():
        # If no file without .z exists, decompression was not successful
        print("[WARNING] Decompression check failed: expected file not found.")
        return False
    return True


def _load_image_data(image: str | Path | np.ndarray, image_size: tuple, color_image: bool) -> np.ndarray:
    """
    Load image and convert it into a coherent size. Returns a numpy array containing the image data.

    Parameters
    ----------
    image : str | Path | np.ndarray
        Path to the image. Can also be a numpy array containing the specific pixels
    image_size : tuple
        images of various sizes can be converted into a coherent size.
        The tuple equals (width, height) of the images.
        Can also be None if the image size should not be changed
    color_image : bool
        Specifies if the loaded image is a color image

    Returns
    -------
    image_data : np.ndarray
        The numpy array containing the image data
    """
    if isinstance(image, (str, PurePath)):
        pil_image = Image.open(image)
    else:
        pil_image = Image.fromarray(np.uint8(image))
    if color_image:
        pil_image = pil_image.convert("RGB")
    # Convert to coherent size
    if image_size is not None:
        pil_image = pil_image.resize(image_size)
    image_data = np.array(pil_image).copy()
    pil_image.close()
    assert image_size is None or image_data.shape == (
        image_size[0], image_size[1], 3), "Size of image is not correct. Should be {0} but is {1}".format(image_size,
                                                                                                          image_data.shape)
    return image_data


class _StemmedCountVectorizer(CountVectorizer):
    """
    Helper class to apply the stemming when counting words in a corpus. Combines the sklearn CountVectorizer with the nltk SnowballStemmer.
    See: https://stackoverflow.com/questions/36182502/add-stemming-support-to-countvectorizer-sklearn
    """

    def build_analyzer(self):
        """
        Custom build_analyzer method. Calls the build_analyzer of the CountVectorizer parent class and then applies
        SnowballStemmer('english')

        Returns
        -------
        stemmed_words : Generator
            the stemmed words in the document
        """
        stemmer = SnowballStemmer('english')
        analyzer = super(_StemmedCountVectorizer, self).build_analyzer()
        stemmed_words = lambda doc: (stemmer.stem(word) for word in analyzer(doc))
        return stemmed_words


def _transform_text_data(data: np.ndarray, use_tfidf: bool, use_stemming: bool, use_stop_words: bool, max_df: float | int, 
                         min_df: float | int, max_features: int, min_variance : float, sublinear_tf: bool, 
                         data_all: np.ndarray | None = None) -> tuple[np.ndarray, list[str]]:
    """
    Transform a set of texts into a data matrix.
    Result can be either a raw count matrix or the result of tf-idf.
    The pipeline is: creation of the count matrix -> (optional) remove words/features with low variance -> (optional) apply tf-idf

    Parameters
    ----------
    data : np.ndarray
        The given data set containing the raw texts
    use_tfidf : bool
        If true, tf-idf will be applied as the last step of the pipeline
    use_stemming : bool
        If true, the SnowballStemmer from nltk will be used when creating the count matrix
    use_stop_words : bool
        If true, the list of English stopwords from sklearn CountVectorizer will be used
    max_df : float | int
        Ignore words that have a document frequency strictly higher than max_df. 
        If float, the parameter represents a proportion of documents, integer corresponds to absolute counts (see sklearn CountVectorizer)
    min_df : float | int
        Ignore words that have a document frequency strictly lower than min_df.
        If float, the parameter represents a proportion of documents, integer corresponds to absolute counts (see sklearn CountVectorizer)
    max_features : int
        If not None, the resulting count matric will ony contain the top max_features ordered by term frequency across the corpus (see sklearn CountVectorizer).
        Note that this value could be further reduced if min_variance is smaller than one
    min_variance : float
        Features with a variance lower than min_variance will be removed (see sklearn VarianceThreshold). 
        The default is to keep all features with non-zero variance, i.e. remove only the features that have the same value in all samples 
    sublinear_tf : bool
        Apply sublinear term frequency scaling, i.e. replace tf with 1 + log(tf) (see sklearn TfidfTransformer)
    data_all : np.ndarray | None
        The complete data set, i.e., if no subset is used. If it is None, it will be equal to data (default: None)

    Returns
    -------
    tuple : tuple[np.ndarray, list[str]]
        The resulting data array,
        The vocabulary of the data output
    """
    if data_all is None:
        data_all = data
    # Create count matrix
    if use_stemming:
        vectorizer = _StemmedCountVectorizer(dtype=np.float64, stop_words="english" if use_stop_words else None, min_df=min_df, max_df=max_df, max_features=max_features)
    else:
        vectorizer = CountVectorizer(dtype=np.float64, stop_words="english" if use_stop_words else None, min_df=min_df, max_df=max_df, max_features=max_features)
    data_sparse_all = vectorizer.fit_transform(data_all)
    data_sparse = vectorizer.transform(data)
    vocabulary = vectorizer.get_feature_names_out()
    # (Optional) Check for variance threshold
    if min_variance != 0:
        selector = VarianceThreshold(min_variance)
        data_sparse_all = selector.fit_transform(data_sparse_all)
        data_sparse = selector.transform(data_sparse)
        vocabulary_mask = selector._get_support_mask()
        vocabulary = vocabulary[vocabulary_mask]
    # (Optional) Apply tf-idf
    if use_tfidf:
        tfidf = TfidfTransformer(sublinear_tf=sublinear_tf)
        tfidf.fit(data_sparse_all)
        data_sparse = tfidf.transform(data_sparse)
    data = np.asarray(data_sparse.todense())
    return data, vocabulary


def flatten_images(data: np.ndarray, format: str) -> np.ndarray:
    """
    Convert data array from image to numerical vector.
    Before flattening, color images will be converted to the HWC/HWDC (height, width, color channels) format.

    Parameters
    ----------
    data : np.ndarray
        The given data set
    format : str
        Format of the images with the data array. Can be: "HW", "HWD", "CHW", "CHWD", "HWC", "HWDC".
        Abbreviations stand for: H: Height, W: Width, D: Depth, C: Color-channels

    Returns
    -------
    data : np.ndarray
        The flatten data array
    """
    format_possibilities = ["HW", "HWD", "CHW", "CHWD", "HWC", "HWDC"]
    assert format in format_possibilities, "Format must be within {0}".format(format_possibilities)
    if format == "HW":
        assert data.ndim == 3, f"ndim has to be 3 but is {data.ndim}"
    elif format in ["HWD", "CHW", "HWC"]:
        assert data.ndim == 4, f"ndim has to be 4 but is {data.ndim}"
    elif format in ["CHWD", "HWDC"]:
        assert data.ndim == 5, f"ndim has to be 5 but is {data.ndim}"
    # Flatten shape
    if format != "HW" and format != "HWD":
        if format == "CHW":
            # Change representation to HWC
            data = np.transpose(data, [0, 2, 3, 1])
        elif format == "CHWD":
            # Change representation to HWDC
            data = np.transpose(data, [0, 2, 3, 4, 1])
        assert data.shape[
                   -1] == 3, "Color-channels must be in the last position and contain three channels not {0} ({1})".format(
            data.shape[-1], data.shape)
    data = data.reshape(data.shape[0], -1)
    return data


def unflatten_images(data_flatten: np.ndarray, image_size: tuple) -> np.ndarray:
    """
    Convert data array from numerical vector to image.
    After unflattening, color images will be converted to the CHW/CHWD (color channels, height, width) format.

    Parameters
    ----------
    data_flatten : np.ndarray
        The given flatten data set
    image_size : str
        The size of a single image, e.g., (28,28,3) for a colored image of size 28 x 28

    Returns
    -------
    data_image : np.ndarray
        The unflatten data array corresponding to an image
    """
    new_shape = tuple([-1] + [i for i in image_size])
    data_image = data_flatten.reshape(new_shape)
    # Change image from HWC/HWDC to CHW/CHWD
    if data_image.ndim == 4 and image_size[-1] == 3:
        data_image = np.transpose(data_image, (0, 3, 1, 2))
    elif data_image.ndim == 5 and image_size[-1] == 3:
        data_image = np.transpose(data_image, (0, 4, 1, 2, 3))
    return data_image

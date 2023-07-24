import numpy as np
import urllib.request
import requests
import os
from pathlib import Path
import ssl

DEFAULT_DOWNLOAD_PATH = str(Path.home() / "Downloads/clustpy_datafiles")


def _get_download_dir(downloads_path: str) -> str:
    """
    Helper function to define the path where the data files should be stored. If downloads_path is None then default path
    '[USER]/Downloads/clustpy_datafiles' will be used. If the directory does not exists it will be created.

    Parameters
    ----------
    downloads_path : str
        path to the directory where the data will be stored. Can be None

    Returns
    -------
    downloads_path : str
        path to the directory where the data will be stored. If input was None this will be equal to
        '[USER]/Downloads/clustpy_datafiles'
    """
    if downloads_path is None:
        downloads_path = DEFAULT_DOWNLOAD_PATH
    if not os.path.isdir(downloads_path):
        os.makedirs(downloads_path)
        with open(downloads_path + "/info.txt", "w") as f:
            f.write("This directory was created by the ClustPy python package to store real world data sets.\n"
                    "The default directory is '[USER]/Downloads/clustpy_datafiles' and can be changed with the "
                    "'downloads_path' parameter when loading a data set.")
    return downloads_path


def _download_file(file_url: str, filename_local: str) -> None:
    """
    Helper function to download a file into a specified location.

    Parameters
    ----------
    file_url : str
        URL of the file
    filename_local : str
        local name of the file after it has been downloaded
    """
    print("Downloading data set from {0} to {1}".format(file_url, filename_local))
    default_ssl = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    urllib.request.urlretrieve(file_url, filename_local)
    ssl._create_default_https_context = default_ssl


def _download_file_from_google_drive(file_id: str, filename_local: str, chunk_size: int = 32768) -> None:
    """
    Download a file from google drive.
    Code taken from:
    https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url

    Parameters
    ----------
    file_id : str
        ID of the file on google drive
    filename_local : str
        local name of the file after it has been downloaded
    chunk_size : int
        chink size when downloading the file (default: 32768)
    """
    print("Downloading data set {0} from Google Drive to {1}".format(file_id, filename_local))
    URL = "https://docs.google.com/uc?export=download&confirm=1"
    session = requests.Session()
    response = session.get(URL, params={"id": file_id, "confirm": 1}, stream=True)
    with open(filename_local, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def _load_data_file(filename_local: str, file_url: str, delimiter: str = ",", last_column_are_labels: bool = True) -> (
        np.ndarray, np.ndarray):
    """
    Helper function to load a data file. Either the first or last column, depending on last_column_are_labels, of the
    data file is used as the label column.
    If file does not exist on the local machine it will be downloaded.

    Parameters
    ----------
    filename_local : str
        local name of the file after it has been downloaded
    file_url : str
        URL of the file
    delimiter : str
        delimiter in the data file (default: ";")
    last_column_are_labels : bool
        specifies if the last column contains the labels. If false labels should be contained in the first column (default: True)

    Returns
    -------
    data, labels : (np.ndarray, np.ndarray)
        the data numpy array, the labels numpy array
    """
    if not os.path.isfile(filename_local):
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


def _decompress_z_file(filename: str, directory: str) -> bool:
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
    os.system("7z x {0} -o{1}".format(filename.replace("\\", "/"), directory.replace("\\", "/")))
    successful = True
    if not os.path.isfile(filename[:-2]):
        # If no file without .z exists, decompression was not successful
        successful = False
        print("[WARNING] 7Zip is needed to uncompress *.Z files!")
    return successful

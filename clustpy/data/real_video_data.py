from clustpy.data._utils import _download_file, _get_download_dir
import cv2
import numpy as np
import os
import zipfile
import torch
from clustpy.data.real_torchvision_data import _torch_normalize_and_flatten


def _load_video(path: str, black_and_white: bool = False) -> np.ndarray:
    """
    Load a video by saving each frame within a numpy array.

    Parameters
    ----------
    path : str
        Path to the video
    black_and_white : bool
        Defines if it is a black and white or a color video

    Returns
    -------
    array : np.ndarray
        The numpy array containing the frames
    """
    # Load video
    vid = cv2.VideoCapture(path)
    # Get parameters from video
    n_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Create numpy array
    array = np.empty((n_frames, height, width, 1 if black_and_white else 3), np.dtype('uint8'))
    # Iterate over frames
    frame_i = 0
    successful = True
    while successful and frame_i < n_frames:
        successful, array[frame_i] = vid.read()
        frame_i += 1
    if frame_i != n_frames:
        print(
            "[WARNING] Not all frames from {0} have been loaded. Number of loaded frames = {1}, total number of frames = {2}".format(
                path, frame_i, n_frames))
    vid.release()
    return array


def load_video_weizmann(flatten: bool = True, normalize_channels: bool = False, downloads_path: str = None) -> (
        np.ndarray, np.ndarray):
    """
    Load the Weizmann video data set.
    It consists of 93 videos showing 9 different persons performing 10 different activities.
    We transform the data set by extracting the 5687 144x180 colored frames.
    The two label sets are the activities and the persons.
    N=5687, d=77760, k=[10, 9].

    Parameters
    ----------
    flatten : bool
        should the image data be flatten, i.e. should the format be changed to a (N x d) array.
        If false, the image will be returned in the CHW format (default: True)
    normalize_channels : bool
        normalize each color-channel of the images (default: False)
    downloads_path : str
        path to the directory where the data is stored (default: None -> [USER]/Downloads/clustpy_datafiles)

    Returns
    -------
    data, labels : (np.ndarray, np.ndarray)
        the data numpy array (5687 x 77760), the labels numpy array (5687 x 2)

    References
    -------
    https://www.wisdom.weizmann.ac.il/~vision/SpaceTimeActions.html
    """
    directory = _get_download_dir(downloads_path) + "/Video_Weizmann/"
    all_actions = ["walk", "run", "jump", "side", "bend", "wave1", "wave2", "pjump", "jack", "skip"]
    all_persons = ["daria", "denis", "eli", "ido", "ira", "lena", "lyova", "moshe", "shahar"]
    all_data = np.zeros((0, 144, 180, 3), dtype="uint8")
    labels = np.zeros((0, 2), dtype="int32")
    # Download data
    for action in all_actions:
        my_zip_file = action + ".zip"
        filename = directory + my_zip_file
        if not os.path.isfile(filename):
            if not os.path.isdir(directory):
                os.mkdir(directory)
            _download_file(
                "https://www.wisdom.weizmann.ac.il/~vision/VideoAnalysis/Demos/SpaceTimeActions/DB/" + my_zip_file,
                filename)
            # Unpack zipfile
            with zipfile.ZipFile(filename, 'r') as zipf:
                zipf.extractall(directory)
    # Load data
    for file in os.listdir(directory):
        if file.endswith(".avi"):
            data_local = _load_video(directory + "/" + file)
            all_data = np.append(all_data, data_local, axis=0)
            # Get name of person and type of activity
            relevant_parts = file.split(".")[0]
            person = relevant_parts.split("_")[0]
            action = relevant_parts.split("_")[1]
            if not action.startswith("wave") and (action.endswith("1") or action.endswith("2")):
                action = action[:-1]
            assert person in all_persons, "Wrong person. {0} is unknown".format(person)
            assert action in all_actions, "Wrong action. {0} is unknown".format(action)
            # Transform string to label
            person_label = all_persons.index(person)
            action_label = all_actions.index(action)
            labels_local = np.array([[action_label, person_label]] * data_local.shape[0], dtype="int32")
            labels = np.append(labels, labels_local, axis=0)
    # If desired, normalize channels
    data_torch = torch.Tensor(all_data)
    is_color_channel_last = True
    data_torch = _torch_normalize_and_flatten(data_torch, flatten, normalize_channels, is_color_channel_last)
    # Move data to CPU
    data = data_torch.detach().cpu().numpy()
    return data, labels

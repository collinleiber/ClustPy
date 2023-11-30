from clustpy.data._utils import _download_file, _get_download_dir, _load_image_data
from clustpy.data.real_torchvision_data import _torch_normalize_and_flatten
import cv2
import numpy as np
import os
import zipfile
import torch

"""
Helpers
"""


def _load_video(path: str, image_size: tuple) -> np.ndarray:
    """
    Load a video by saving each frame within a numpy array.

    Parameters
    ----------
    path : str
        Path to the video
    image_size : tuple
        The single frames can be downsized. This is necessary for large datasets.
        The tuple equals (width, height) of the images.
        Can also be None if the image size should not be changed

    Returns
    -------
    video_array : np.ndarray
        The array containing the frames
    """
    # Load video
    vid = cv2.VideoCapture(path)
    video_array = []
    # Iterate over frames
    successful = True
    while successful:
        successful, frame_array = vid.read()
        if successful:
            is_color_image = frame_array.ndim == 3 and frame_array.shape[2] == 3
            if is_color_image:
                frame_array = cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB)
            if image_size is not None:
                frame_array = _load_image_data(frame_array, image_size, is_color_image)
            video_array.append(frame_array)
    vid.release()
    # Transform list to numpy array
    video_array = np.array(video_array, dtype="uint8")
    return video_array


def _downsample_frames(data: np.ndarray, labels: np.ndarray, frame_sampling_ratio: float = 1) -> (
        np.ndarray, np.ndarray):
    """
    Downsample the number of frames within a video.

    Parameters
    ----------
    data : np.ndarray
        The data array containing the frames
    labels : np.ndarray
        The labels array
    frame_sampling_ratio : float
        Ratio to downsample the number of frames. If it is set to 1 all frames will be returned.
        Can take values within (0, 1] (default: 1)

    Returns
    -------
    data, labels : (np.ndarray, np.ndarray)
        The updated data array, the updated labels array
    """
    assert frame_sampling_ratio > 0 and frame_sampling_ratio <= 1, "frame_sampling_ratio must be within (0, 1]"
    # Downsample array
    if frame_sampling_ratio != 1:
        n_samples_orig = data.shape[0]
        n_to_delete = int(n_samples_orig - frame_sampling_ratio * n_samples_orig)
        indices_to_delete = np.round(np.linspace(0, n_samples_orig - 1, n_to_delete)).astype(int)
        data = np.delete(data, indices_to_delete, axis=0)
        labels = np.delete(labels, indices_to_delete, axis=0)
        assert frame_sampling_ratio <= data.shape[
            0] / n_samples_orig, "Difference between frame_sampling_ratio ({0}) and actual sampling ratio ({1}) is too large".format(
            frame_sampling_ratio, data.shape[0] / n_samples_orig)
    return data, labels


"""
Actual datasets
"""


def load_video_weizmann(image_size: tuple = None, frame_sampling_ratio: float = 1, flatten: bool = True,
                        normalize_channels: bool = False, downloads_path: str = None) -> (np.ndarray, np.ndarray):
    """
    Load the Weizmann video data set.
    It consists of 93 videos showing 9 different persons performing 10 different activities.
    We transform the data set by extracting the 5687 144x180 colored frames.
    The two label sets are the activities and the persons.
    N=5687, d=77760, k=[10, 9].

    Parameters
    ----------
    image_size : tuple
        The single frames can be downsized. This is necessary for large datasets.
        The tuple equals (width, height) of the images.
        Can also be None if the image size should not be changed (default: None)
    frame_sampling_ratio : float
        Ratio to downsample the number of frames of each video. If it is set to 1 all frames will be returned.
        Can take values within (0, 1] (default: 1)
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
    all_data = np.zeros(
        (0, 144 if image_size is None else image_size[0], 180 if image_size is None else image_size[1], 3),
        dtype="uint8")
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
    # Load data, iterate over all video files
    for v_file in os.listdir(directory):
        # Ignore zip files
        if v_file.endswith(".avi"):
            data_local = _load_video(directory + "/" + v_file, image_size)
            # Get name of person and type of activity
            relevant_parts = v_file.split(".")[0]
            person = relevant_parts.split("_")[0]
            action = relevant_parts.split("_")[1]
            # Sometimes a person performs an action twice. In that case a 1/2 is appended to the action
            if not action.startswith("wave") and (action.endswith("1") or action.endswith("2")):
                action = action[:-1]
            assert person in all_persons, "Wrong person. {0} is unknown".format(person)
            assert action in all_actions, "Wrong action. {0} is unknown".format(action)
            # Transform string to label
            label_person = all_persons.index(person)
            label_action = all_actions.index(action)
            labels_local = np.array([[label_action, label_person]] * data_local.shape[0], dtype="int32")
            # Downsample frames
            data_local, labels_local = _downsample_frames(data_local, labels_local, frame_sampling_ratio)
            # Update data and labels
            all_data = np.append(all_data, data_local, axis=0)
            labels = np.append(labels, labels_local, axis=0)
    # If desired, normalize channels
    data_torch = torch.Tensor(all_data)
    is_color_channel_last = True
    data_torch = _torch_normalize_and_flatten(data_torch, flatten, normalize_channels, is_color_channel_last)
    # Move data to CPU
    data = data_torch.detach().cpu().numpy()
    return data, labels


def load_video_keck_gesture(subset: str = "all", image_size: tuple = (200, 200), frame_sampling_ratio: float = 1,
                            flatten: bool = True, normalize_channels: bool = False, downloads_path: str = None) -> (
        np.ndarray, np.ndarray):
    """
    Load the Keck Gesture video data set.
    It consists of 42 training and 56 testing videos showing 4 different persons performing 14 different gestures.
    We assign the label '0' to the gesture 'no gesture', which describes the frames between the actual gestures.
    This results in 15 different gestures.
    Note, that the person with label '3' is only contained in the testing data.
    We transform the data set by extracting the 25457 480x640 colored frames.
    Further, we recommend to downsize the frames due to possible memory issues.
    The final data set is divided into 13546 training and 11911 test images.
    The two label sets are the gestures and the persons.
    N=25457, d=120000 (for image_size (200, 200)), k=[15, 4].

    Parameters
    ----------
    subset : str
        can be 'all', 'test' or 'train'. 'all' combines test and train data (default: 'all')
    image_size : tuple
        The single frames can be downsized. This is necessary for large datasets.
        The tuple equals (width, height) of the images.
        Can also be None if the image size should not be changed (default: (200, 200)))
    frame_sampling_ratio : float
        Ratio to downsample the number of frames of each video. If it is set to 1 all frames will be returned.
        Can take values within (0, 1] (default: 1)
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
        the data numpy array (25457 x 120000 (for image_size (200, 200))), the labels numpy array (25457 x 2)

    References
    -------
    http://www.zhuolin.umiacs.io/Keckgesturedataset.html
    """

    def parse_frames_file(frames_file: str) -> (dict, dict):
        """
        Get the specific frames for each gesture from the frames.txt.

        Parameters
        ----------
        frames_file : str
            path to the frames txt.

        Returns
        -------
        train_dict, test_dict : (dict, dict)
            The dictionary for the training data, the dictionary for the testing data
        """
        train_dict = {}
        test_dict = {}
        train_data = True
        with open(frames_file, "r") as f:
            all_lines = f.readlines()
            for line in all_lines:
                # Read infos from file
                if line.startswith("person"):
                    person = int(line.split("_")[0].replace("person ", "")) - 1
                    gesture = int(line.split("_")[1].replace("gesture", ""))
                    frame_limits = line.split("frames ")[1].split(",")
                    frames = [(int(single_limit.split("-")[0]), int(single_limit.split("-")[1]) + 1) for single_limit in
                              frame_limits]
                    if train_data:
                        # Train data entry
                        train_dict[(gesture, person)] = frames
                    else:
                        # Test data entry
                        test_dict[(gesture, person)] = frames
                # Switch to test data
                if line.startswith("Testing set:"):
                    train_data = False
        return train_dict, test_dict

    # Start loading the dataset
    subset = subset.lower()
    assert subset in ["all", "train",
                      "test"], "subset must match 'all', 'train' or 'test'. Your input {0}".format(subset)
    directory = _get_download_dir(downloads_path) + "/Video_Keck_Gesture/"
    filename = directory + "Keck_Dataset.zip"
    frames_file = directory + "sequences.txt"
    if not os.path.isfile(filename):
        if not os.path.isdir(directory):
            os.mkdir(directory)
        _download_file("http://www.zhuolin.umiacs.io/PrototypeTree/Keck_Dataset.zip", filename)
        # Unpack zipfile
        with zipfile.ZipFile(filename, 'r') as zipf:
            zipf.extractall(directory)
        # Get Relevant frames
        _download_file("http://www.zhuolin.umiacs.io/PrototypeTree/sequences.txt", frames_file)
    # Load data and labels
    all_data = np.zeros(
        (0, 480 if image_size is None else image_size[0], 640 if image_size is None else image_size[1], 3),
        dtype="uint8")
    labels = np.zeros((0, 2), dtype="int32")
    # Get frame limits from sequences file
    frames_train_dict, frames_test_dict = parse_frames_file(frames_file)
    # Get necessary directories
    file_directories = []
    if subset == "all" or subset == "train":
        file_directories.append((True, "training files/"))
    if subset == "all" or subset == "test":
        file_directories.append((False, "testingfiles/"))
    # load videos
    for train_data, file_directory in file_directories:
        directory_files = directory + "Keck Dataset/" + file_directory
        # Iterate over all video files
        for v_file in os.listdir(directory_files):
            data_local = _load_video(directory_files + v_file, image_size)
            # Transform string to label
            label_gesture = int(v_file.split("_")[1].replace("gesture", ""))
            label_person = int(v_file.split("_")[0].replace("person", "")) - 1
            labels_local = np.array([[0, label_person]] * data_local.shape[0], dtype="int32")
            # Use frames_dicts to set gestures correctly
            if train_data:
                for start, end in frames_train_dict[(label_gesture, label_person)]:
                    labels_local[start:end, 0] = label_gesture
            else:
                for start, end in frames_test_dict[(label_gesture, label_person)]:
                    labels_local[start:end, 0] = label_gesture
            # Downsample frames
            data_local, labels_local = _downsample_frames(data_local, labels_local, frame_sampling_ratio)
            # Update data and labels
            all_data = np.append(all_data, data_local, axis=0)
            labels = np.append(labels, labels_local, axis=0)
    # If desired, normalize channels
    data_torch = torch.Tensor(all_data)
    is_color_channel_last = True
    data_torch = _torch_normalize_and_flatten(data_torch, flatten, normalize_channels, is_color_channel_last)
    # Move data to CPU
    data = data_torch.detach().cpu().numpy()
    return data, labels

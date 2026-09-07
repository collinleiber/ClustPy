import json
import numpy as np
import pytest
import inspect
from sklearn.utils import Bunch

import clustpy
import clustpy.data._cache as cache_module
from clustpy.data._cache import cache_dataset, clear_cache, clear_dataset_cache, USE_CACHE_DEFAULT

# ============================================================================
# Test helpers
# ============================================================================


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    """
    Use an isolated temporary cache directory for every test.
    """
    monkeypatch.setattr(cache_module, "CACHE_DIR", tmp_path / "clustpy")

    return cache_module.CACHE_DIR


def make_bunch(
    dataset_name="TestDataset",
    value=1,
):
    """
    Create a small deterministic Bunch for testing.
    """
    data = np.array(
        [
            [value, value + 1],
            [value + 2, value + 3],
        ]
    )

    target = np.array([0, 1])

    return Bunch(
        dataset_name=dataset_name,
        data=data,
        target=target,
    )


# ============================================================================
# Basic caching behavior
# ============================================================================


def test_cache_dataset_returns_bunch(cache_dir):
    """
    The decorated function should return its original Bunch.
    """

    calls = 0

    @cache_dataset
    def load_test(return_X_y=False, use_cache=True):
        nonlocal calls
        calls += 1

        return make_bunch()

    result = load_test()

    assert isinstance(result, Bunch)
    assert result.dataset_name == "TestDataset"
    assert np.array_equal(result.data, np.array([[1, 2], [3, 4]]))
    assert np.array_equal(result.target, np.array([0, 1]))

    assert calls == 1


def test_cache_is_used_on_second_call(cache_dir):
    """
    The underlying function must only be called once when the same
    dataset is requested twice.
    """

    calls = 0

    @cache_dataset
    def load_test(return_X_y=False, use_cache=True):
        nonlocal calls
        calls += 1

        return make_bunch()

    first = load_test(use_cache=True)
    second = load_test()

    assert calls == 1

    assert np.array_equal(first.data, second.data)
    assert np.array_equal(first.target, second.target)
    assert first.dataset_name == second.dataset_name


def test_cache_file_is_created(cache_dir):
    """
    A pickle file and metadata JSON file must be created.
    """

    @cache_dataset
    def load_test(return_X_y=False, use_cache=True):
        return make_bunch()

    load_test()

    cache_files = list(cache_dir.rglob("*"))

    pickle_files = [file for file in cache_files if file.suffix == ".pkl"]

    json_files = [file for file in cache_files if file.suffix == ".json"]

    assert len(pickle_files) == 1
    assert len(json_files) == 1


def test_cache_directory_uses_function_name(cache_dir):
    """
    Cache files should be stored below a directory named after the
    dataset loader function.
    """

    @cache_dataset
    def load_my_dataset(return_X_y=False, use_cache=True):
        return make_bunch()

    load_my_dataset()

    expected_directory = cache_dir / "load_my_dataset"

    assert expected_directory.exists()
    assert expected_directory.is_dir()


# ============================================================================
# use_cache
# ============================================================================


def test_use_cache_false_bypasses_cache(cache_dir):
    """
    use_cache=False must always execute the underlying function.
    """

    calls = 0

    @cache_dataset
    def load_test(return_X_y=False, use_cache=True):
        nonlocal calls
        calls += 1

        return make_bunch(value=calls)

    first = load_test(use_cache=False)
    second = load_test(use_cache=False)

    assert calls == 2

    assert np.array_equal(
        first.data,
        np.array([[1, 2], [3, 4]]),
    )

    assert np.array_equal(
        second.data,
        np.array([[2, 3], [4, 5]]),
    )


def test_use_cache_false_does_not_create_cache(cache_dir):
    """
    Calling a loader with use_cache=False must not create cache files.
    """

    @cache_dataset
    def load_test(return_X_y=False, use_cache=True):
        return make_bunch()

    load_test(use_cache=False)

    assert not cache_dir.exists()


def test_default_use_cache(cache_dir):
    """
    The current implementation defaults to use_cache=USE_CACHE_DEFAULT.
    """

    calls = 0

    @cache_dataset
    def load_test(return_X_y=False, use_cache=USE_CACHE_DEFAULT):
        nonlocal calls
        calls += 1

        return make_bunch()

    load_test()
    load_test()

    if USE_CACHE_DEFAULT:
        assert calls == 1
    else:
        assert calls == 2


# ============================================================================
# return_X_y
# ============================================================================


def test_return_X_y_true_returns_tuple(cache_dir):
    """
    return_X_y=True must return (data, target).
    """

    @cache_dataset
    def load_test(return_X_y=False, use_cache=True):
        return make_bunch()

    data, target = load_test(return_X_y=True)

    assert isinstance(data, np.ndarray)
    assert isinstance(target, np.ndarray)

    assert np.array_equal(
        data,
        np.array([[1, 2], [3, 4]]),
    )

    assert np.array_equal(
        target,
        np.array([0, 1]),
    )


def test_return_X_y_true_still_caches_bunch(cache_dir):
    """
    Even if the caller requests return_X_y=True, the underlying function
    must be called with return_X_y=False so that a Bunch is cached.
    """

    calls = 0
    received_return_X_y = []

    @cache_dataset
    def load_test(return_X_y=False, use_cache=True):
        nonlocal calls

        calls += 1
        received_return_X_y.append(return_X_y)

        if return_X_y:
            # This intentionally fails if the decorator does not force
            # return_X_y=False.
            return (
                np.array([[1, 2], [3, 4]]),
                np.array([0, 1]),
            )

        return make_bunch()

    data, target = load_test(return_X_y=True)

    assert calls == 1
    assert received_return_X_y == [False]

    assert np.array_equal(
        data,
        np.array([[1, 2], [3, 4]]),
    )

    assert np.array_equal(
        target,
        np.array([0, 1]),
    )


def test_return_X_y_does_not_create_separate_cache(cache_dir):
    """
    return_X_y=True and return_X_y=False must use the same cache entry.
    """

    calls = 0

    @cache_dataset
    def load_test(return_X_y=False, use_cache=True):
        nonlocal calls
        calls += 1

        return make_bunch()

    bunch = load_test(return_X_y=False)
    data, target = load_test(return_X_y=True)

    assert calls == 1

    assert np.array_equal(bunch.data, data)
    assert np.array_equal(bunch.target, target)

    pickle_files = list(cache_dir.rglob("*.pkl"))

    assert len(pickle_files) == 1


# ============================================================================
# Cache key / parameters
# ============================================================================


def test_different_parameters_create_different_cache_entries(cache_dir):
    """
    Dataset-defining parameters must be part of the cache key.
    """

    calls = 0

    @cache_dataset
    def load_test(size=1, return_X_y=False, use_cache=True):
        nonlocal calls
        calls += 1

        return make_bunch(value=size)

    first = load_test(size=1)
    second = load_test(size=2)

    assert calls == 2

    assert not np.array_equal(first.data, second.data)

    pickle_files = list(cache_dir.rglob("*.pkl"))

    assert len(pickle_files) == 2


def test_same_parameters_reuse_cache(cache_dir):
    """
    Identical dataset-defining parameters must produce the same cache entry.
    """

    calls = 0

    @cache_dataset
    def load_test(size=1, return_X_y=False, use_cache=True):
        nonlocal calls
        calls += 1

        return make_bunch(value=size)

    load_test(size=10)
    load_test(size=10)

    assert calls == 1

    pickle_files = list(cache_dir.rglob("*.pkl"))

    assert len(pickle_files) == 1


def test_parameter_order_does_not_change_cache_key(cache_dir):
    """
    Keyword argument ordering must not create separate cache entries.
    """

    calls = 0

    @cache_dataset
    def load_test(a=1, b=2, return_X_y=False, use_cache=True):
        nonlocal calls
        calls += 1

        return make_bunch(value=a + b)

    load_test(a=1, b=2)
    load_test(b=2, a=1)

    assert calls == 1

    pickle_files = list(cache_dir.rglob("*.pkl"))

    assert len(pickle_files) == 1


def test_list_and_tuple_parameters_are_supported(cache_dir):
    """
    Lists and tuples should result in deterministic cache keys.
    """

    calls = 0

    @cache_dataset
    def load_test(features=(1, 2), return_X_y=False, use_cache=True):
        nonlocal calls
        calls += 1

        return make_bunch(value=sum(features))

    load_test(features=[1, 2])
    load_test(features=[1, 2])

    assert calls == 1


def test_dictionary_parameters_are_supported(cache_dir):
    """
    Dictionaries should be converted into deterministic representations.
    """

    calls = 0

    @cache_dataset
    def load_test(options=None, return_X_y=False, use_cache=True):
        nonlocal calls
        calls += 1

        value = options["a"] + options["b"]

        return make_bunch(value=value)

    load_test(options={"a": 1, "b": 2})
    load_test(options={"b": 2, "a": 1})

    assert calls == 1


def test_path_parameters_are_supported(cache_dir, tmp_path):
    """
    Path objects should be handled correctly when they participate in
    the cache key.
    """

    calls = 0

    @cache_dataset
    def load_test(path=None, return_X_y=False, use_cache=True):
        nonlocal calls
        calls += 1

        return make_bunch()

    load_test(path=tmp_path / "data")
    load_test(path=tmp_path / "data")

    assert calls == 1


# ============================================================================
# Ignored parameters
# ============================================================================


def test_downloads_path_is_ignored(cache_dir, tmp_path):
    """
    downloads_path must not influence the cache key.
    """

    calls = 0

    @cache_dataset
    def load_test(
        downloads_path=None,
        return_X_y=False,
        use_cache=True,
    ):
        nonlocal calls
        calls += 1

        return make_bunch()

    path1 = tmp_path / "downloads1"
    path2 = tmp_path / "downloads2"

    load_test(downloads_path=path1)
    load_test(downloads_path=path2)

    assert calls == 1

    pickle_files = list(cache_dir.rglob("*.pkl"))

    assert len(pickle_files) == 1


def test_return_X_y_is_ignored_in_cache_key(cache_dir):
    """
    return_X_y must not influence the cache key.
    """

    calls = 0

    @cache_dataset
    def load_test(return_X_y=False, use_cache=True):
        nonlocal calls
        calls += 1

        return make_bunch()

    load_test(return_X_y=False)
    load_test(return_X_y=True)

    assert calls == 1

    pickle_files = list(cache_dir.rglob("*.pkl"))

    assert len(pickle_files) == 1


# ============================================================================
# Versioning
# ============================================================================


def test_clustpy_version_is_part_of_cache_key(cache_dir, monkeypatch):
    """
    Changing the ClustPy version must invalidate the cache.
    """

    calls = 0

    @cache_dataset
    def load_test(return_X_y=False, use_cache=True):
        nonlocal calls
        calls += 1

        return make_bunch(value=calls)

    load_test()

    assert calls == 1

    monkeypatch.setattr(clustpy, "__version__", "999.999.999")

    result = load_test()

    assert calls == 2
    assert result.data[0, 0] == 2

    pickle_files = list(cache_dir.rglob("*.pkl"))

    assert len(pickle_files) == 2


def test_metadata_contains_clustpy_version(cache_dir):
    """
    Cache metadata must contain the current ClustPy version.
    """

    @cache_dataset
    def load_test(return_X_y=False, use_cache=True):
        return make_bunch()

    load_test()

    metadata_files = list(cache_dir.rglob("*.json"))

    assert len(metadata_files) == 1

    with metadata_files[0].open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    assert metadata["clustpy_version"] == clustpy.__version__


# ============================================================================
# Metadata
# ============================================================================


def test_metadata_contains_expected_information(cache_dir):
    """
    The JSON metadata should contain all important cache information.
    """

    @cache_dataset
    def load_test(
        size=42,
        return_X_y=False,
        use_cache=True,
    ):
        return make_bunch(value=size)

    load_test(size=42)

    metadata_files = list(cache_dir.rglob("*.json"))

    assert len(metadata_files) == 1

    with metadata_files[0].open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    assert metadata["dataset_name"] == "TestDataset"
    assert metadata["clustpy_version"] == clustpy.__version__

    assert metadata["function"].endswith("load_test")

    assert metadata["parameters"]["size"] == 42

    assert "cache_key" in metadata
    assert isinstance(metadata["cache_key"], str)
    assert len(metadata["cache_key"]) == 64


def test_metadata_and_pickle_have_same_cache_key(cache_dir):
    """
    The cache filename and metadata must reference the same cache key.
    """

    @cache_dataset
    def load_test(return_X_y=False, use_cache=True):
        return make_bunch()

    load_test()

    pickle_files = list(cache_dir.rglob("*.pkl"))
    metadata_files = list(cache_dir.rglob("*.json"))

    assert len(pickle_files) == 1
    assert len(metadata_files) == 1

    cache_key_from_filename = pickle_files[0].stem

    with metadata_files[0].open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    assert metadata["cache_key"] == cache_key_from_filename


# ============================================================================
# Validation of loader return value
# ============================================================================


def test_missing_dataset_name_raises_error(cache_dir):
    """
    A decorated loader must return an object with dataset_name.
    """

    @cache_dataset
    def load_test(return_X_y=False, use_cache=True):
        return Bunch(
            data=np.array([[1, 2]]),
            target=np.array([0]),
        )

    with pytest.raises(TypeError, match="dataset_name"):
        load_test()


def test_missing_data_raises_error(cache_dir):
    """
    A decorated loader must return an object containing data.
    """

    @cache_dataset
    def load_test(return_X_y=False, use_cache=True):
        return Bunch(
            dataset_name="TestDataset",
            target=np.array([0]),
        )

    with pytest.raises(TypeError, match="data"):
        load_test()


def test_missing_target_raises_error(cache_dir):
    """
    A decorated loader must return an object containing target.
    """

    @cache_dataset
    def load_test(return_X_y=False, use_cache=True):
        return Bunch(
            dataset_name="TestDataset",
            data=np.array([[1, 2]]),
        )

    with pytest.raises(TypeError, match="target"):
        load_test()


# ============================================================================
# Test Cache Decorator Signature and Wrapper Preservation
# ============================================================================


def test_cache_dataset_preserves_original_signature():
    def example_loader(
        dataset_name="test",
        return_X_y=False,
        downloads_path=None,
    ):
        pass

    decorated_loader = cache_dataset(example_loader)

    signature = inspect.signature(decorated_loader)

    assert "dataset_name" in signature.parameters
    assert "return_X_y" in signature.parameters
    assert "downloads_path" in signature.parameters

    assert signature.parameters["dataset_name"].default == "test"
    assert signature.parameters["return_X_y"].default is False
    assert signature.parameters["downloads_path"].default is None


def test_cache_dataset_exposes_expected_signature():
    def example_loader(
        dataset_name="test",
        return_X_y=False,
        downloads_path=None,
    ):
        pass

    decorated_loader = cache_dataset(example_loader)

    signature = inspect.signature(decorated_loader)
    parameters = list(signature.parameters.values())

    assert parameters == [
        inspect.Parameter(
            "dataset_name",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default="test",
        ),
        inspect.Parameter(
            "return_X_y",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=False,
        ),
        inspect.Parameter(
            "downloads_path",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=None,
        ),
    ]


def test_cache_dataset_preserves_different_loader_signatures():
    def example_loader(
        subset="train",
        ignore_small_clusters=False,
        image_size=(200, 200),
        frame_sampling_ratio=0.5,
        return_X_y=False,
        downloads_path=None,
        use_cache=True,
    ):
        pass

    original_signature = inspect.signature(example_loader)
    decorated_loader = cache_dataset(example_loader)
    decorated_signature = inspect.signature(decorated_loader)

    # All original parameters must be preserved exactly
    for parameter_name, parameter in original_signature.parameters.items():
        assert parameter_name in decorated_signature.parameters
        assert decorated_signature.parameters[parameter_name] == parameter

    # Check the individual parameter types/defaults
    assert decorated_signature.parameters["subset"].default == "train"
    assert decorated_signature.parameters["ignore_small_clusters"].default is False
    assert decorated_signature.parameters["image_size"].default == (200, 200)
    assert decorated_signature.parameters["frame_sampling_ratio"].default == 0.5
    assert decorated_signature.parameters["return_X_y"].default is False
    assert decorated_signature.parameters["downloads_path"].default is None
    assert decorated_signature.parameters["use_cache"].default is True


def test_cache_dataset_preserves_complex_loader_signature():
    def example_loader(
        subset="all",
        ignore_small_clusters=True,
        image_size=(200, 200),
        frame_sampling_ratio=0.5,
        return_X_y=False,
        downloads_path=None,
    ):
        pass

    original_signature = inspect.signature(example_loader)
    decorated_loader = cache_dataset(example_loader)
    decorated_signature = inspect.signature(decorated_loader)

    # The original signature must be preserved
    assert list(decorated_signature.parameters) == [
        "subset",
        "ignore_small_clusters",
        "image_size",
        "frame_sampling_ratio",
        "return_X_y",
        "downloads_path",
    ]

    for parameter_name, parameter in original_signature.parameters.items():
        assert decorated_signature.parameters[parameter_name] == parameter

    # Verify different kinds of defaults
    assert decorated_signature.parameters["subset"].default == "all"
    assert decorated_signature.parameters["ignore_small_clusters"].default is True
    assert decorated_signature.parameters["image_size"].default == (200, 200)
    assert decorated_signature.parameters["frame_sampling_ratio"].default == 0.5


def test_cache_dataset_preserves_wrapped_function():
    def example_loader(dataset_name="test"):
        """Example loader documentation."""
        pass

    decorated_loader = cache_dataset(example_loader)

    assert decorated_loader.__wrapped__ is example_loader
    assert decorated_loader.__name__ == example_loader.__name__
    assert decorated_loader.__module__ == example_loader.__module__
    assert decorated_loader.__doc__ == example_loader.__doc__


def test_cache_dataset_preserves_getfullargspec():
    def example_loader(
        subset="train",
        ignore_small_clusters=False,
        image_size=(200, 200),
        frame_sampling_ratio=0.5,
        return_X_y=False,
        downloads_path=None,
    ):
        pass

    original_args = inspect.getfullargspec(example_loader).args
    decorated_loader = cache_dataset(example_loader)
    decorated_args = inspect.getfullargspec(decorated_loader).args

    # The positional/original arguments must remain unchanged.
    assert decorated_args == original_args


@pytest.mark.parametrize(
    "loader_name",
    [
        "load_soybean_large",
    ],
)
def test_cache_dataset_preserves_real_loader_getfullargspec(
    loader_name,
):
    import clustpy.data as data

    loader = getattr(data, loader_name)
    original_signature = inspect.getfullargspec(loader)

    decorated_loader = cache_dataset(loader)
    decorated_signature = inspect.getfullargspec(decorated_loader)

    # Positional arguments are preserved
    assert decorated_signature.args == original_signature.args

    # *args is preserved
    assert decorated_signature.varargs == original_signature.varargs

    # **kwargs is preserved
    assert decorated_signature.varkw == original_signature.varkw

    # Keyword-only arguments are preserved
    assert decorated_signature.kwonlyargs == original_signature.kwonlyargs

    # Keyword-only defaults are preserved
    assert decorated_signature.kwonlydefaults == original_signature.kwonlydefaults


# ============================================================================
# Cache clearing
# ============================================================================


def test_clear_dataset_cache(cache_dir):
    """
    clear_dataset_cache() should remove all cache entries belonging to
    the specified dataset.
    """

    @cache_dataset
    def load_test_a(return_X_y=False, use_cache=True):
        return make_bunch(dataset_name="DatasetA")

    @cache_dataset
    def load_test_b(return_X_y=False, use_cache=True):
        return make_bunch(dataset_name="DatasetB")

    load_test_a()
    load_test_b()

    assert len(list(cache_dir.rglob("*.pkl"))) == 2
    assert len(list(cache_dir.rglob("*.json"))) == 2

    deleted = clear_dataset_cache("DatasetA")

    assert deleted == 1

    remaining_metadata = list(cache_dir.rglob("*.json"))

    assert len(remaining_metadata) == 1

    with remaining_metadata[0].open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    assert metadata["dataset_name"] == "DatasetB"


def test_clear_dataset_cache_only_deletes_matching_dataset(cache_dir):
    """
    Clearing one dataset must not affect other datasets.
    """

    @cache_dataset
    def load_dataset_a(return_X_y=False, use_cache=True):
        return make_bunch(dataset_name="DatasetA")

    @cache_dataset
    def load_dataset_b(return_X_y=False, use_cache=True):
        return make_bunch(dataset_name="DatasetB")

    load_dataset_a()
    load_dataset_b()

    clear_dataset_cache("DatasetA")

    assert len(list(cache_dir.rglob("*.pkl"))) == 1

    metadata_files = list(cache_dir.rglob("*.json"))

    assert len(metadata_files) == 1

    with metadata_files[0].open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    assert metadata["dataset_name"] == "DatasetB"


def test_clear_dataset_cache_nonexistent_dataset(cache_dir):
    """
    Clearing a dataset that does not exist should do nothing.
    """

    @cache_dataset
    def load_test(return_X_y=False, use_cache=True):
        return make_bunch()

    load_test()

    deleted = clear_dataset_cache("DoesNotExist")

    assert deleted == 0

    assert len(list(cache_dir.rglob("*.pkl"))) == 1


def test_clear_cache_removes_all_datasets(cache_dir):
    """
    clear_cache() should remove all cached datasets.
    """

    @cache_dataset
    def load_dataset_a(return_X_y=False, use_cache=True):
        return make_bunch(dataset_name="DatasetA")

    @cache_dataset
    def load_dataset_b(return_X_y=False, use_cache=True):
        return make_bunch(dataset_name="DatasetB")

    load_dataset_a()
    load_dataset_b()

    assert cache_dir.exists()

    deleted = clear_cache()

    # Each dataset has one pickle and one JSON file.
    assert deleted == 4

    assert not cache_dir.exists()


def test_clear_cache_when_cache_does_not_exist(cache_dir):
    """
    clear_cache() should safely handle a missing cache directory.
    """

    assert not cache_dir.exists()

    deleted = clear_cache()

    assert deleted is None


def test_clear_dataset_cache_when_cache_does_not_exist(cache_dir):
    """
    clear_dataset_cache() should safely handle a missing cache directory.
    """

    assert not cache_dir.exists()

    deleted = clear_dataset_cache("DoesNotExist")

    assert deleted is None


# ============================================================================
# Multiple cache entries for one dataset
# ============================================================================


def test_multiple_parameter_variants_can_be_cleared(cache_dir):
    """
    clear_dataset_cache() must remove all parameter variants of one dataset.
    """

    @cache_dataset
    def load_test(size=1, return_X_y=False, use_cache=True):
        return make_bunch(
            dataset_name="TestDataset",
            value=size,
        )

    load_test(size=1)
    load_test(size=2)
    load_test(size=3)

    assert len(list(cache_dir.rglob("*.pkl"))) == 3
    assert len(list(cache_dir.rglob("*.json"))) == 3

    deleted = clear_dataset_cache("TestDataset")

    assert deleted == 3

    assert not list(cache_dir.rglob("*.pkl"))
    assert not list(cache_dir.rglob("*.json"))


# ============================================================================
# Cache persistence
# ============================================================================


def test_cached_bunch_is_loaded_from_disk(cache_dir):
    """
    Verify that the second call really reads the serialized Bunch rather
    than retaining an in-memory object.
    """

    calls = 0

    @cache_dataset
    def load_test(return_X_y=False, use_cache=True):
        nonlocal calls
        calls += 1

        return make_bunch()

    first = load_test()

    assert calls == 1

    # Mutate the first returned object.
    first.data[0, 0] = 999

    second = load_test()

    # The second object should have been deserialized from disk.
    assert calls == 1
    assert second.data[0, 0] == 1


def test_cached_bunch_is_independent_from_original(cache_dir):
    """
    Mutating the originally returned Bunch must not modify the cached copy.
    """

    @cache_dataset
    def load_test(return_X_y=False, use_cache=True):
        return make_bunch()

    result = load_test()

    result.data[:] = 999
    result.target[:] = 999

    cached_result = load_test()

    assert np.array_equal(
        cached_result.data,
        np.array([[1, 2], [3, 4]]),
    )

    assert np.array_equal(
        cached_result.target,
        np.array([0, 1]),
    )


# ============================================================================
# Decorator metadata / function behavior
# ============================================================================


def test_wraps_preserves_function_name(cache_dir):
    """
    functools.wraps should preserve the original function name.
    """

    @cache_dataset
    def load_my_dataset(return_X_y=False, use_cache=True):
        return make_bunch()

    assert load_my_dataset.__name__ == "load_my_dataset"


def test_wraps_preserves_docstring(cache_dir):
    """
    functools.wraps should preserve the original docstring.
    """

    @cache_dataset
    def load_my_dataset(return_X_y=False, use_cache=True):
        """My test dataset."""
        return make_bunch()

    assert load_my_dataset.__doc__ == "My test dataset."


# ============================================================================
# Real ClustPy loader integration test
# ============================================================================


def test_real_loader_load_iris_uses_cache(cache_dir):
    """
    Integration test using an actual ClustPy dataset loader.

    load_iris is deliberately used because it is small and does not require
    a large external dataset download.
    """

    from clustpy.data import load_iris

    first = load_iris(use_cache=True)

    assert isinstance(first, Bunch)
    assert hasattr(first, "dataset_name")
    assert hasattr(first, "data")
    assert hasattr(first, "target")

    pickle_files = list(cache_dir.rglob("*.pkl"))
    metadata_files = list(cache_dir.rglob("*.json"))

    assert len(pickle_files) == 1
    assert len(metadata_files) == 1

    second = load_iris(use_cache=True)

    assert isinstance(second, Bunch)

    assert np.array_equal(first.data, second.data)
    assert np.array_equal(first.target, second.target)


def test_real_loader_load_iris_return_X_y(cache_dir):
    """
    Actual loader should correctly return X and y from the cached Bunch.
    """

    from clustpy.data import load_iris

    bunch = load_iris(use_cache=True)

    data, target = load_iris(
        return_X_y=True,
        use_cache=True,
    )

    assert np.array_equal(data, bunch.data)
    assert np.array_equal(target, bunch.target)

    # Both calls must use exactly one cache entry.
    assert len(list(cache_dir.rglob("*.pkl"))) == 1

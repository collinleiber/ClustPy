import clustpy
import hashlib
import inspect
import json
import pickle
from functools import wraps
from pathlib import Path
from typing import Any

# Default cache directory
CACHE_DIR = Path.home() / ".cache" / "clustpy"

USE_CACHE_DEFAULT = True


def _make_cache_key_value(value: Any) -> Any:
    """
    Convert common Python objects into deterministic representations.

    This is mainly needed for parameters such as lists, tuples, dictionaries,
    Paths, numpy values, etc.
    """
    if isinstance(value, dict):
        return tuple(sorted((key, _make_cache_key_value(val)) for key, val in value.items()))

    if isinstance(value, (list, tuple)):
        return tuple(_make_cache_key_value(item) for item in value)

    if isinstance(value, set):
        return tuple(sorted(_make_cache_key_value(item) for item in value))

    if isinstance(value, Path):
        return str(value)

    return value


def _make_json_serializable(value: Any) -> Any:
    """
    Convert common Python objects into JSON-serializable representations.

    Used only for storing cache metadata in the accompanying JSON file.
    """
    if isinstance(value, dict):
        return {str(key): _make_json_serializable(val) for key, val in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_make_json_serializable(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    # Handle common numpy scalar types without requiring numpy as a dependency
    # of this utility.
    if hasattr(value, "item") and callable(value.item):
        try:
            return value.item()
        except (ValueError, TypeError):
            pass

    # JSON already knows how to serialize these
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    # Fallback for other objects
    return repr(value)


def _create_cache_key(
    function_name: str,
    parameters: dict[str, Any],
) -> str:
    """
    Create a deterministic cache key based on the function, ClustPy version,
    and dataset-defining parameters.
    """
    cache_data = {
        "clustpy_version": clustpy.__version__,
        "function": function_name,
        "parameters": _make_cache_key_value(parameters),
    }

    serialized = pickle.dumps(cache_data)

    return hashlib.sha256(serialized).hexdigest()


def cache_dataset(func):
    """
    Decorator for caching ClustPy dataset loading functions.

    The decorated function must return a sklearn Bunch containing at least
    the attributes `dataset_name`, `data`, and `target`.

    Parameters named `use_cache`, `return_X_y`, and `downloads_path` are not
    considered part of the cache key.

    When `use_cache=True`, the wrapped function is always executed with
    `return_X_y=False` so that the Bunch can be cached. If the caller
    requested `return_X_y=True`, `(data, target)` is returned afterwards.
    """

    signature = inspect.signature(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Bind arguments to their parameter names
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()

        # Check for `use_cache` in the call
        # if not existent there -> check in the func's arguments
        # if also not existent there -> use `USE_CACHE_DEFAULT`
        use_cache = kwargs.get("use_cache")
        if use_cache == None:
            use_cache = bound.arguments.get("use_cache")
        if use_cache == None:
            use_cache = USE_CACHE_DEFAULT

        # No caching requested
        if not use_cache:
            return func(*args, **kwargs)

        # Remember what the caller requested
        return_X_y = bound.arguments.get("return_X_y", False)

        # Parameters that don't define the actual dataset
        ignored_parameters = {
            "return_X_y",
            "use_cache",
            "downloads_path",
        }

        cache_parameters = {name: value for name, value in bound.arguments.items() if name not in ignored_parameters}

        # Function name
        function_name = func.__module__ + "." + func.__qualname__

        # Create cache key
        cache_key = _create_cache_key(
            function_name=function_name,
            parameters=cache_parameters,
        )

        # Cache directory
        #
        # We don't know Bunch.dataset_name until the function has
        # been executed once, therefore use the function name as
        # the cache namespace.
        #
        cache_dir = CACHE_DIR / func.__name__
        cache_dir.mkdir(parents=True, exist_ok=True)

        cache_file = cache_dir / f"{cache_key}.pkl"
        metadata_file = cache_dir / f"{cache_key}.json"

        # Try loading cached Bunch
        if cache_file.exists():
            with cache_file.open("rb") as file:
                bunch = pickle.load(file)

        else:
            # Force return_X_y=False for the actual loader call
            function_arguments = dict(bound.arguments)
            function_arguments["return_X_y"] = False

            # use_cache is not passed to the original function
            function_arguments.pop("use_cache", None)

            bunch = func(**function_arguments)

            # Sanity check
            if not hasattr(bunch, "dataset_name"):
                raise TypeError(
                    f"{func.__name__}() returned an object without "
                    "'dataset_name'. A Bunch is required when using "
                    "@cache_dataset."
                )

            if not hasattr(bunch, "data") or not hasattr(bunch, "target"):
                raise TypeError(f"{func.__name__}() returned an object without " "'data' and/or 'target'.")

            # Store Bunch
            with cache_file.open("wb") as file:
                pickle.dump(
                    bunch,
                    file,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            # Store metadata
            metadata = {
                "dataset_name": bunch.dataset_name,
                "clustpy_version": clustpy.__version__,
                "function": function_name,
                "parameters": _make_json_serializable(cache_parameters),
                "cache_key": cache_key,
            }

            with metadata_file.open("w", encoding="utf-8") as file:
                json.dump(
                    metadata,
                    file,
                    indent=4,
                    ensure_ascii=False,
                )

        # Convert to (X, y) if requested
        if return_X_y:
            return bunch.data, bunch.target

        bunch.cached = True
        bunch.cache_path = cache_file
        bunch.cache_metadata_path = metadata_file
        return bunch

    # Copy functions signature to wrapper
    wrapper.__signature__ = signature
    return wrapper


def clear_dataset_cache(dataset_name: str):
    """
    Delete the cache of a specific dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset, e.g. "Banknotes" or "VideoWeizmann".
    """

    # Find cache directories whose metadata contains this dataset name.
    if not CACHE_DIR.exists():
        return

    deleted_files = 0

    for cache_dir in CACHE_DIR.iterdir():

        if not cache_dir.is_dir():
            continue

        for metadata_file in cache_dir.glob("*.json"):

            try:
                with metadata_file.open("r", encoding="utf-8") as file:
                    metadata = json.load(file)
            except (OSError, json.JSONDecodeError):
                continue

            if metadata.get("dataset_name") != dataset_name:
                continue

            cache_key = metadata.get("cache_key")

            if cache_key is None:
                continue

            cache_file = cache_dir / f"{cache_key}.pkl"

            # Delete Bunch
            if cache_file.exists():
                cache_file.unlink()
                deleted_files += 1

            # Delete metadata
            metadata_file.unlink()

    # Remove empty cache directories
    for cache_dir in CACHE_DIR.iterdir():
        if cache_dir.is_dir() and not any(cache_dir.iterdir()):
            cache_dir.rmdir()

    return deleted_files


def clear_cache():
    """
    Delete all cached datasets.
    """
    if not CACHE_DIR.exists():
        return

    deleted_files = 0

    for cache_dir in CACHE_DIR.iterdir():

        if not cache_dir.is_dir():
            continue

        for cache_file in cache_dir.iterdir():
            if cache_file.is_file():
                cache_file.unlink()
                deleted_files += 1

        # Remove now-empty dataset directory
        if not any(cache_dir.iterdir()):
            cache_dir.rmdir()

    # Remove root cache directory if empty
    if CACHE_DIR.exists() and not any(CACHE_DIR.iterdir()):
        CACHE_DIR.rmdir()

    return deleted_files

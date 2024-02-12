from clustpy.data import ZNormalizer, z_normalization
import numpy as np


def _check_normalized(data, feature_or_channel_wise, should_be_normalized):
    imprecision = 1e-5
    # Check is simple if we only have a single channel, i.e. a grayscale image
    if not feature_or_channel_wise:
        if should_be_normalized:
            assert abs(np.mean(data) - 0) < imprecision
            assert abs(np.std(data) - 1) < imprecision
        else:
            assert abs(np.mean(data) - 0) > imprecision
            assert abs(np.std(data) - 1) > imprecision
    else:
        # Else we have to check each channel separately
        for i in range(data.shape[1]):
            if should_be_normalized:
                assert abs(np.mean(data[:, i]) - 0) < imprecision
                assert abs(np.std(data[:, i]) - 1) < imprecision
            else:
                assert abs(np.mean(data[:, i]) - 0) > imprecision
                assert abs(np.std(data[:, i]) - 1) > imprecision


def test_ZNormalizer_tabular_data():
    # Create data
    rs = np.random.RandomState(1)
    data = np.c_[rs.uniform(low=-5, high=10, size=(1000, 5)), rs.uniform(low=-20, high=20, size=(1000, 5))]
    assert data.shape == (1000, 10)
    _check_normalized(data, False, False) # not normalized regarding all features
    _check_normalized(data, True, False) # not normalized regarding feature-wise
    # Check normalization regarding all data
    normalizer = ZNormalizer(feature_or_channel_wise=False)
    data_z = normalizer.fit_transform(data)
    _check_normalized(data_z, False, True) # normalized regarding all features
    _check_normalized(data_z, True, False) # not normalized regarding feature-wise
    assert data.shape == data_z.shape
    assert np.array_equal(data_z, z_normalization(data, feature_or_channel_wise=False))
    assert np.allclose(data, normalizer.inverse_transform(data_z))
    # Check normalization feature-wise
    normalizer = ZNormalizer(feature_or_channel_wise=True)
    data_z = normalizer.fit_transform(data)
    _check_normalized(data_z, True, True) # normalized regarding feature-wise
    assert data.shape == data_z.shape
    assert np.array_equal(data_z, z_normalization(data, feature_or_channel_wise=True))
    assert np.allclose(data, normalizer.inverse_transform(data_z))


def test_ZNormalizer_grayscale_images():
    # Create data
    rs = np.random.RandomState(1)
    data = np.r_[rs.uniform(low=-5, high=10, size=(500, 10, 10)), rs.uniform(low=-20, high=20, size=(500, 10, 10))]
    assert data.shape == (1000, 10, 10)
    _check_normalized(data, False, False) # not normalized regarding all features
    _check_normalized(data, True, False) # not normalized regarding feature-wise
    # Check normalization regarding all data
    normalizer = ZNormalizer(feature_or_channel_wise=False)
    data_z = normalizer.fit_transform(data)
    _check_normalized(data_z, False, True) # normalized regarding all features
    _check_normalized(data_z, True, False) # not normalized regarding feature-wise
    assert data.shape == data_z.shape
    assert np.array_equal(data_z, z_normalization(data, feature_or_channel_wise=False))
    assert np.allclose(data, normalizer.inverse_transform(data_z))
    # Check normalization feature-wise
    normalizer = ZNormalizer(feature_or_channel_wise=True)
    data_z2 = normalizer.fit_transform(data)
    _check_normalized(data_z, False, True) # normalized regarding feature-wise
    assert np.allclose(data_z, data_z2)


def test_ZNormalizer_color_images():
    # Create data
    rs = np.random.RandomState(1)
    data = np.r_[
        rs.uniform(low=-5, high=10, size=(500, 3, 10, 10)), rs.uniform(low=-20, high=20, size=(500, 3, 10, 10))]
    assert data.shape == (1000, 3, 10, 10)
    _check_normalized(data, False, False) # not normalized regarding all features
    _check_normalized(data, True, False) # not normalized regarding feature-wise
    # Check normalization regarding all data
    normalizer = ZNormalizer(feature_or_channel_wise=False)
    data_z = normalizer.fit_transform(data)
    _check_normalized(data_z, False, True) # normalized regarding all features
    _check_normalized(data_z, True, False) # not normalized regarding feature-wise
    assert data.shape == data_z.shape
    assert np.array_equal(data_z, z_normalization(data, feature_or_channel_wise=False))
    assert np.allclose(data, normalizer.inverse_transform(data_z))
    # Check normalization feature-wise
    normalizer = ZNormalizer(feature_or_channel_wise=True)
    data_z = normalizer.fit_transform(data)
    _check_normalized(data_z, True, True)  # normalized regarding feature-wise
    assert data.shape == data_z.shape
    assert np.array_equal(data_z, z_normalization(data, feature_or_channel_wise=True))
    assert np.allclose(data, normalizer.inverse_transform(data_z))

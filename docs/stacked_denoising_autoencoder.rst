Stacked Denoising Autoencoder
===============

To use a stacked denoising autoencoder [1], we first create an ordinary stacked autoencoder (SAE).
For our example we use the well-known MNIST [2] dataset.

.. code-block:: python
    from clustpy.deep.neural_networks import StackedAutoencoder
    from clustpy.data import load_mnist
    import torch

    data, labels = load_mnist(return_X_y=True)
    data = torch.from_numpy(data).float()
    SAE = StackedAutoencoder(layers=[data.shape[1], 256, 128, 64, 10])

In this example, the SAE has three hidden layers with the sizes 256, 128, and 64.
The resulting embedding has 10 features.
Now we could already train the SAE using the default parameters.
However, the data is usually normalized beforehand.

.. code-block:: python
    data_mean = data.mean()
    data_std = data.std()
    data = (data - data_mean) / data_std

We also want to train the SAE with denoising in mind.
In other words, we need a suitable corruption function.
In our case, we choose simple salt and pepper noise.
Note that because the data has been normalized, it does not lie within [0, 1] or [0, 255].
Our corruption function must take this into account.

.. code-block:: python
    data_min = data.min()
    data_max = data.max()

    def my_corruption(data, data_min, data_max, amount_noise=0.02):
        apply_noise = torch.rand(data.shape)
        data[apply_noise < amount_noise] = data_max
        data[apply_noise > 1 - amount_noise] = data_min
        return data

    corruption_fn = lambda data: my_corruption(data, data_min=data_min, data_max=data_max)

Now that we have a suitable corruption function, let us look at its effect regarding a sample.

.. code-block:: python
    from clustpy.utils import plot_image

    sample = data[0].cpu().numpy().reshape((28, 28))
    plot_image(sample, black_and_white=True)
    corrupted_sample = plot_image(corruption_fn(sample), black_and_white=True)

Finally, we can start the actual training of our stacked denoising autoencoder.

.. code-block:: python
    SAE.fit(data=data, corruption_fn=my_corruption)

[1] Vincent, Pascal, et al. "Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion." Journal of machine learning research 11.12 (2010).

[2] LeCun, Yann, et al. "Gradient-based learning applied to document recognition." Proceedings of the IEEE 86.11 (1998): 2278-2324.

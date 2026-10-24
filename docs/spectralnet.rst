SpectralNet: learning an out-of-sample spectral map
===========================================================

``SpectralNet`` implements the unnormalized-Laplacian variant of
`Shaham et al., ICLR 2018 <https://arxiv.org/abs/1801.01587>`_. Unlike a
spectral embedding of a fixed graph, the fitted network can embed new points
without rebuilding that graph.

Training has three stages: an optional autoencoder produces input features;
an optional Siamese network learns a distance from unlabeled neighbor and
non-neighbor pairs; the spectral network alternates orthogonalization and
gradient steps on independently sampled minibatches. KMeans then clusters the
spectral embedding. The output layer stays fixed during prediction, including
when predicting one point at a time.

For example, using a two-dimensional feature array:

.. code-block:: python

   from sklearn.datasets import make_blobs
   from sklearn.model_selection import train_test_split
   from clustpy.deep import SpectralNet

   data, _ = make_blobs(600, n_features=10, centers=3, random_state=0)
   train, test = train_test_split(data, test_size=0.2, random_state=0)
   clustering = SpectralNet(
       n_clusters=3,
       batch_size=256,
       spectral_hidden_layers=(64, 64),
       siamese_hidden_layers=(64, 16),
       siamese_epochs=50,
       clustering_epochs=100,
       random_state=0,
   )
   labels = clustering.fit_predict(train)
   test_embedding = clustering.transform(test)
   test_labels = clustering.predict(test)

``embedding_size`` controls the optional autoencoder's code dimension.
``transform`` instead returns the spectral embedding, with ``n_clusters``
columns. To use the same feature-network configuration as other ClustPy deep
clustering algorithms, pass ``neural_network`` and/or ``neural_network_weights``.
Set ``use_autoencoder=True`` to create a default autoencoder instead. A supplied
feature network is frozen during
spectral optimization. Set ``use_siamese=False`` to use Euclidean affinities
directly.

Choosing the graph and training configuration
---------------------------------------------

Affinities are formed within each minibatch, so both ``batch_size`` and
``n_neighbors`` affect the graph. Too few neighbors can split the data into
more graph components than the intended clusters; increasing only the number
of training epochs cannot remove this ambiguity. Larger batches preserve more
of the neighborhood structure, while affinity storage grows quadratically in
the batch size. No full training-set affinity matrix is stored.
The Gaussian bandwidth is estimated once from training-set neighbor distances
in the selected metric space and remains fixed across minibatches. Set
``affinity_scale`` to supply a positive bandwidth explicitly.

The spectral loss is minimized under an orthogonality constraint. A small
loss by itself does not establish good clustering, particularly when the
affinity graph is fragmented. Numerically rank-deficient network outputs
raise a ``ValueError`` instead of silently accepting a collapsed map. Reducing
the learning rate or changing the architecture may be necessary. The current
implementation uses fixed epoch budgets and configurable RMSprop parameters;
the conservative default spectral learning rate is ``1e-5``.
It does not reproduce the paper's dataset-specific learning-rate searches or
claim its reported MNIST/Reuters scores.

Inputs are two-dimensional feature arrays. Image arrays must be flattened or
encoded before fitting. The selected PyTorch device must support QR and SVD decompositions.

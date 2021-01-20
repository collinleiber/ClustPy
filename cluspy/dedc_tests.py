from cluspy.evaluation import *
from cluspy.preprocessing import preprocess_features
from cluspy.centroid import XMeans, GMeans, PGMeans, DipMeans, ProjectedDipMeans
from sklearn.metrics import normalized_mutual_info_score as nmi, adjusted_rand_score as ari
from cluspy.deep import DEC, DCN, IDEC, DEDC, VaDE
from cluspy.deep._utils import detect_device, get_trained_autoencoder, encode_batchwise
import torch
from cluspy.data import load_har, load_usps, load_mnist, load_fmnist, load_kmnist, \
    load_letterrecognition, load_optdigits, load_pendigits


def preprocess_with_autoencoder(X):
    device = detect_device()
    batch_size = 256
    learning_rate = 1e-3
    pretrain_epochs = 100
    embedding_size = min(X.shape[1], 5)
    optimizer_class = torch.optim.Adam
    loss_fn = torch.nn.MSELoss()
    trainloader = torch.utils.data.DataLoader(torch.from_numpy(X).float(),
                                              batch_size=batch_size,
                                              # sample random mini-batches from the data
                                              shuffle=True,
                                              drop_last=False)
    testloader = torch.utils.data.DataLoader(torch.from_numpy(X).float(),
                                             batch_size=batch_size,
                                             # Note that we deactivate the shuffling
                                             shuffle=False,
                                             drop_last=False)
    autoencoder = get_trained_autoencoder(trainloader, learning_rate, pretrain_epochs, device,
                                          optimizer_class, loss_fn, X.shape[1], embedding_size)
    embedded_data = encode_batchwise(testloader, autoencoder, device)
    return embedded_data


def znorm(X):
    return (X - np.mean(X)) / np.std(X)


def get_subset_mnist(n_clusters):
    X, labels = load_mnist()
    selection = labels < n_clusters
    X = X[selection]
    labels = labels[selection]
    return X, labels


def get_subset_mnist_2():
    return get_subset_mnist(2)

def get_subset_mnist_3():
    return get_subset_mnist(3)

def get_subset_mnist_4():
    return get_subset_mnist(4)

def get_subset_mnist_5():
    return get_subset_mnist(5)

def get_subset_mnist_6():
    return get_subset_mnist(6)

def get_subset_mnist_7():
    return get_subset_mnist(7)

def get_subset_mnist_8():
    return get_subset_mnist(8)

def get_subset_mnist_9():
    return get_subset_mnist(9)

def robustness_test_mnist():
    datasets = [
        EvaluationDataset("MNIST-2", data=get_subset_mnist_2, preprocess_methods=znorm),
        EvaluationDataset("MNIST-3", data=get_subset_mnist_3, preprocess_methods=znorm),
        EvaluationDataset("MNIST-4", data=get_subset_mnist_4, preprocess_methods=znorm),
        EvaluationDataset("MNIST-5", data=get_subset_mnist_5, preprocess_methods=znorm),
        EvaluationDataset("MNIST-6", data=get_subset_mnist_6, preprocess_methods=znorm),
        EvaluationDataset("MNIST-7", data=get_subset_mnist_7, preprocess_methods=znorm),
        EvaluationDataset("MNIST-8", data=get_subset_mnist_8, preprocess_methods=znorm),
        EvaluationDataset("MNIST-9", data=get_subset_mnist_9, preprocess_methods=znorm),
    ]
    algorithms = [EvaluationAlgorithm("DEDC", DEDC, {"n_clusters_start": 35, "batch_size": 256, "pretrain_epochs": 100,
                                                     "dedc_epochs": 50, "embedding_size": 5})
                  ]
    metrics = [EvaluationMetric("NMI", nmi), EvaluationMetric("ARI", ari)]
    df = evaluate_multiple_datasets(datasets, algorithms, metrics, 10, True, True, False, True,
                                    save_path="robustness_test_mnist.csv", save_intermediate_results=True)
    print(df)


def other_robustness_tests():
    datasets = [
        EvaluationDataset("HAR", data=load_har),
        EvaluationDataset("MNIST", data=load_mnist, preprocess_methods=znorm)
    ]
    algorithms = [EvaluationAlgorithm("DEDC_15", DEDC,
                                      {"n_clusters_start": 15, "batch_size": 256, "pretrain_epochs": 100,
                                        "dedc_epochs": 50, "embedding_size": 5}),
                  EvaluationAlgorithm("DEDC_20", DEDC,
                                      {"n_clusters_start": 20, "batch_size": 256, "pretrain_epochs": 100,
                                       "dedc_epochs": 50, "embedding_size": 5}),
                  EvaluationAlgorithm("DEDC_25", DEDC,
                                      {"n_clusters_start": 25, "batch_size": 256, "pretrain_epochs": 100,
                                       "dedc_epochs": 50, "embedding_size": 5}),
                  EvaluationAlgorithm("DEDC_30", DEDC,
                                      {"n_clusters_start": 30, "batch_size": 256, "pretrain_epochs": 100,
                                       "dedc_epochs": 50, "embedding_size": 5}),
                    EvaluationAlgorithm("DEDC_35", DEDC,
                                        {"n_clusters_start": 35, "batch_size": 256, "pretrain_epochs": 100,
                                        "dedc_epochs": 50, "embedding_size": 5}),
                  EvaluationAlgorithm("DEDC_40", DEDC,
                                      {"n_clusters_start": 40, "batch_size": 256, "pretrain_epochs": 100,
                                       "dedc_epochs": 50, "embedding_size": 5}),
                  EvaluationAlgorithm("DEDC_45", DEDC,
                                      {"n_clusters_start": 45, "batch_size": 256, "pretrain_epochs": 100,
                                       "dedc_epochs": 50, "embedding_size": 5}),
                  EvaluationAlgorithm("DEDC_50", DEDC,
                                      {"n_clusters_start": 50, "batch_size": 256, "pretrain_epochs": 100,
                                       "dedc_epochs": 50, "embedding_size": 5})
                  ]
    metrics = [EvaluationMetric("NMI", nmi), EvaluationMetric("ARI", ari)]
    df = evaluate_multiple_datasets(datasets, algorithms, metrics, 10, True, True, False, True,
                                    save_path="robustness_test_mnist.csv", save_intermediate_results=True)
    print(df)


def full_test():
    ignore_deep = ["DEDC", "DEC", "IDEC", "DCN", "VaDE"]
    datasets = [
        EvaluationDataset("MNIST", data=load_mnist, preprocess_methods=znorm, ignore_algorithms=["DipMeans"]),
        EvaluationDataset("AE+MNIST", data=load_mnist, preprocess_methods=[znorm, preprocess_with_autoencoder],
                          ignore_algorithms=ignore_deep),
        EvaluationDataset("FMNIST", data=load_fmnist, preprocess_methods=znorm, ignore_algorithms=["DipMeans"]),
        EvaluationDataset("AE+FMNIST", data=load_fmnist, preprocess_methods=[znorm, preprocess_with_autoencoder],
                          ignore_algorithms=ignore_deep),
        EvaluationDataset("KMNIST", data=load_kmnist, preprocess_methods=znorm, ignore_algorithms=["DipMeans"]),
        EvaluationDataset("AE+KMNIST", data=load_kmnist, preprocess_methods=[znorm, preprocess_with_autoencoder],
                          ignore_algorithms=ignore_deep),
        EvaluationDataset("HAR", data=load_har),
        EvaluationDataset("AE+HAR", data=load_har, preprocess_methods=preprocess_with_autoencoder,
                          ignore_algorithms=ignore_deep),
        EvaluationDataset("USPS", data=load_usps, preprocess_methods=znorm),
        EvaluationDataset("AE+USPS", data=load_usps, preprocess_methods=[znorm, preprocess_with_autoencoder],
                          ignore_algorithms=ignore_deep),
        EvaluationDataset("optdigits", data=load_optdigits, preprocess_methods=znorm),
        EvaluationDataset("AE+optdigits", data=load_optdigits, preprocess_methods=[znorm, preprocess_with_autoencoder],
                          ignore_algorithms=ignore_deep),
        EvaluationDataset("pendigits", data=load_pendigits, preprocess_methods=preprocess_features),
        EvaluationDataset("AE+pendigits", data=load_pendigits,
                          preprocess_methods=[preprocess_features, preprocess_with_autoencoder],
                          ignore_algorithms=ignore_deep),
        EvaluationDataset("letterrecognition", data=load_letterrecognition, preprocess_methods=preprocess_features),
        EvaluationDataset("AE+letterrecognition", data=load_letterrecognition,
                          preprocess_methods=[preprocess_features, preprocess_with_autoencoder],
                          ignore_algorithms=ignore_deep)
    ]
    algorithms = [EvaluationAlgorithm("DEDC", DEDC, {"n_clusters_start": 35, "batch_size": 256, "pretrain_epochs": 100,
                                                     "dedc_epochs": 50, "embedding_size": 5}),
                  EvaluationAlgorithm("DEC", DEC, {"n_clusters": None, "batch_size": 256, "pretrain_epochs": 100,
                                                   "dec_epochs": 150, "embedding_size": 10}),
                  EvaluationAlgorithm("IDEC", IDEC, {"n_clusters": None, "batch_size": 256, "pretrain_epochs": 100,
                                                     "dec_epochs": 150, "embedding_size": 10}),
                  EvaluationAlgorithm("DCN", DCN, {"n_clusters": None, "batch_size": 256, "pretrain_epochs": 100,
                                                   "dcn_epochs": 150, "embedding_size": 10}),
                  EvaluationAlgorithm("VaDE", VaDE, {"n_clusters": None, "batch_size": 256, "pretrain_epochs": 100,
                                                     "vade_epochs": 150, "embedding_size": 10}),
                  EvaluationAlgorithm("XMeans", XMeans, {"max_n_clusters": 35}),
                  EvaluationAlgorithm("GMeans", GMeans, {"max_n_clusters": 35}),
                  EvaluationAlgorithm("PGMeans", PGMeans, {"max_n_clusters": 35}),
                  EvaluationAlgorithm("DipMeans", DipMeans, {"max_n_clusters": 35}),
                  EvaluationAlgorithm("ProjDipMeans", ProjectedDipMeans, {"max_n_clusters": 35}),
                  ]
    metrics = [EvaluationMetric("NMI", nmi), EvaluationMetric("ARI", ari)]
    df = evaluate_multiple_datasets(datasets, algorithms, metrics, 10, True, True, False, True,
                                    save_path="full_test.csv", save_intermediate_results=True)
    print(df)

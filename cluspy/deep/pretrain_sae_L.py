import os
import torch.utils.data
import torch
from ourUtils.funct_band_aids import setup_directory
import numpy as np
import torchvision
from ourUtils.dcrs_utils import random_seed
from ourUtils.torch.aes.ae_utils import init_model
from ourUtils.dataset import Datasets, fetch_data_set_by_name
import pandas as pd
from ourUtils.torch.aes.stacked_ae import stacked_ae


def loss_fn(x, y): return torch.mean((x - y) ** 2)


def get_total_loss(ae, dataloader):
    total_loss = 0.0
    i = 0
    for batch in dataloader:
        batch = batch[0].cuda()
        total_loss += loss_fn(batch, ae.forward(batch)[1]).item()
        i += 1
    return total_loss / (i + 1)


def add_noise(batch):
    mask = torch.empty(
        batch.shape, device=batch.device).bernoulli_(0.8)
    return batch * mask


def pretrain():
    nr_aes = 10
    # Nr of Training steps per layer
    steps_per_layer = 25000
    lr = 1e-4
    # Nr of Finetuning
    refine_training_steps = 50000
    embd_sz = 10
    for data_set_name_i in [Datasets.GTSRB]:
        result_dir = os.path.join("results", data_set_name_i)
        setup_directory(result_dir)
        models_dir = os.path.join(result_dir, "pretrained")
        setup_directory(models_dir)
        print("######################")
        print("Load data set: ", data_set_name_i)
        print("######################")
        pt_data, pt_labels = fetch_data_set_by_name(data_set_name_i, train=True, flattened=True)
        np_seeds = np.random.randint(900000, size=nr_aes)
        n_points = pt_data.shape[0]
        n_features = pt_data.shape[1]
        nr_of_clusters = len(set(pt_labels.tolist()))
        print("n_points: ", n_points)
        print("n_features: ", n_features)
        print("nr_of_clusters: ", nr_of_clusters)
        #embd_sz = np.min([n_features, nr_of_clusters])
        print("embd_sz: ", embd_sz)
        rec_losses_pretrained = []
        rec_losses_fine_tuned = []
        for ae_index in range(0, nr_aes):
            print("\nStart training ae {} with random seed {}".format(
                ae_index, np_seeds[ae_index]))
            random_seed(np_seeds[ae_index])
            model_name = f"ae-model-hl-{embd_sz}-idx-{ae_index}.pth"

            train = torch.utils.data.TensorDataset(pt_data)
            train_loader = torch.utils.data.DataLoader(
                train, batch_size=256, shuffle=True, pin_memory=True)

            ae = init_model(n_features, embd_sz, lr=lr).cuda()

            ae.pretrain(train_loader, rounds_per_layer=steps_per_layer,
                        dropout_rate=0.2, corruption_fn=add_noise)

            rec_losses_pretrained.append(get_total_loss(ae, train_loader))

            print(f"Complete data loss after pretraining {rec_losses_pretrained[-1]}")
            # uncomment if you want to save the layer wise pretrained models
            # torch.save(ae.state_dict(), os.path.join(
            #     models_dir, "layer_wise_pretrained_" + model_name))

            ae.refine_training(train_loader, refine_training_steps,
                               corruption_fn=add_noise)

            rec_losses_fine_tuned.append(get_total_loss(ae, train_loader))
            print(f"Complete data loss after fine tuning {rec_losses_fine_tuned[-1]}")
            torch.save(ae.state_dict(), os.path.join(
                models_dir, "fine_tuned_" + model_name))
            print("saved model")
            del ae
        pd.DataFrame({"rec_losses_fine_tuned": rec_losses_fine_tuned,
                      "rec_losses_pretrained": rec_losses_pretrained}).to_csv(
            os.path.join(models_dir, f"scores-hl-{embd_sz}.csv"), sep=";", index=False)


if __name__ == "__main__":
    pretrain()

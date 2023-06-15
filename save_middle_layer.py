
from semi_supervised_bbbc import Semi_supervised_VAE, plot_ELBO, plot_1_reconstruction
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import sys


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 300
epochs = 100
batch_size = 100

classes = 13

input_dim = 68
channels = 3

train_size = 100_000
test_size = 2_000


#latent_dim = 10
#epochs, batch_size, train_size, test_size = 2, 10, 10, 10

torch.backends.cudnn.deterministic = True
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

from dataloader import BBBC

main_path = "/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/"
#main_path = "/Users/nikolaj/Fagprojekt/Data/"


exclude_dmso = False
shuffle = True

subset = (train_size, test_size)

dataset_train = BBBC(folder_path=main_path + "singh_cp_pipeline_singlecell_images",
                        meta_path=main_path + "metadata.csv",
                        subset=subset,
                        test=False,
                        exclude_dmso=exclude_dmso,
                        shuffle=shuffle)

dataset_test = BBBC(folder_path=main_path + "singh_cp_pipeline_singlecell_images",
                        meta_path=main_path + "metadata.csv",
                        subset=subset,
                        test=True,
                        exclude_dmso=exclude_dmso,
                        shuffle=shuffle)


loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, drop_last=True)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, drop_last=True)

VAE = Semi_supervised_VAE(classes=classes, 
                          latent_dim=latent_dim,
                          input_dim=input_dim, 
                          channels=channels,
                          beta=0.1).to(device)


classes, indexes = np.unique(dataset_test.meta[dataset_test.col_names[-1]], return_index=True)
boolean = len(classes) == 13 if not exclude_dmso else len(classes) == 12

if not boolean: 
    #raise Warning("The number of unique drugs in the test set is not 13")
    print("The number of unique drugs in the test set is not 13")

#print("VAE:")
#summary(VAE, input_size=(channels, input_dim, input_dim))

trained_encoder, trained_decoder, trained_classifier, train_REs, train_KLs, train_ELBOs = VAE.train_VAE(
                    dataloader=loader_train, epochs=epochs)


results_folder = 'save_middle/'

if not(os.path.exists(results_folder)):
    os.mkdir(results_folder)


test_REs, test_KLs, test_ELBOs = VAE.test_VAE(dataloader=loader_test)

train_images_folder = results_folder +'train_images/'

test_images_folder = results_folder + 'test_images/'

if not(os.path.exists(train_images_folder)):
    os.mkdir(train_images_folder)

if not(os.path.exists(test_images_folder)):
    os.mkdir(test_images_folder)


np.savez(results_folder + "train_ELBOs.npz", train_ELBOs=train_ELBOs)
np.savez(results_folder + "train_REs.npz", train_REs=train_REs)
np.savez(results_folder + "train_KLs.npz", train_KLs=train_KLs)

np.savez(results_folder + "test_ELBOs.npz", test_ELBOs=test_ELBOs)
np.savez(results_folder + "test_REs.npz", test_REs=test_REs)
np.savez(results_folder + "test_KLs.npz", test_KLs=test_KLs)

# torch.save(encoder_VAE, results_folder + "encoder.pt")
# torch.save(decoder_VAE, results_folder + "decoder.pt")

torch.save(VAE.middel_1, results_folder + "middel_1.pt")
torch.save(VAE.middel_2, results_folder + "middel_2.pt")
torch.save(VAE.middel_3, results_folder + "middel_3.pt")


plot_ELBO(train_REs, train_KLs, train_ELBOs, name="ELBO-components", results_folder=results_folder)
    

for i, image in enumerate(loader_train.dataset):
    if i == 10:
        break
    plot_1_reconstruction(image['image'],
                        vae = VAE,
                        name="Train " + str(image['id']), 
                        results_folder=train_images_folder, 
                        latent_dim=latent_dim, 
                        channels=channels, 
                        input_dim=input_dim)

if boolean:
    for index in indexes:
        image = dataset_test[index]
        plot_1_reconstruction(image['image'],
                            vae = VAE,
                            name=image['moa_name'] + ' ' + str(image['id']), 
                            results_folder=test_images_folder, 
                            latent_dim=latent_dim, 
                            channels=channels, 
                            input_dim=input_dim)

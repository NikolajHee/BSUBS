import os
from dataloader import BBBC
from torch.utils.data import DataLoader
from gaussian_bbbc import VAE, generate_image, log_Normal, log_standard_Normal, encoder, decoder
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
if '___ARKIV' in os.listdir():
    from torchsummary import summary


plt.style.use('fivethirtyeight')

def plot_reconstruction(data, name, encoder_VAE, decoder_VAE, latent_dim, channels, input_dim, results_folder=''):
        generated_images = []
        for i in range(9):
            image = generate_image(
                data[i]['image'],
                encoder_VAE,
                decoder=decoder_VAE,
                latent_dim=latent_dim,
                channels=channels,
                input_dim=input_dim,
            )
            generated_images.append(image)

        fig, ax = plt.subplots(9, 2)
        for i in range(9):
            ax[i, 0].imshow(data[i]['image'].reshape((68,68,3)), cmap="gray")
            ax[i, 0].set_xticks([])
            ax[i, 0].set_yticks([])
            ax[i, 1].imshow(generated_images[i].reshape((68,68,3)), cmap="gray")
            ax[i, 1].set_xticks([])
            ax[i, 1].set_yticks([])
        fig.suptitle(name)
        plt.savefig(results_folder + name + '.png')
        plt.show()


def plot_elbo(reconstruction_errors, regularizers, title, results_folder=''):
    plt.plot(
        np.arange(0, len(reconstruction_errors), 1),
        reconstruction_errors,
        label="Reconstruction Error",
    )
    plt.plot(np.arange(0, len(regularizers), 1), regularizers, label="Regularizer")
    plt.xlabel("iterationns")
    plt.title(title)
    plt.legend()
    plt.savefig(results_folder + title + '.png')
    plt.show()
    

     


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


latent_dims = [2**i for i in range(7,12)]
epochs = 200
batch_size = 40

train_size = 20000
test_size = 1000

# latent_dims = [2**i for i in range(2,4)]
# epochs = 1
# batch_size = 1

# train_size = 10
# test_size = 10


input_dim = 68
channels = 3

subset = (train_size, train_size)


if '___ARKIV' in os.listdir():
    main_path = "/Users/nikolaj/Fagprojekt/Data/"
else:
    main_path = "/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/"
    

folder_path = os.path.join(main_path, "singh_cp_pipeline_singlecell_images")
meta_path= os.path.join(main_path, "metadata.csv")

trainset = BBBC(folder_path=folder_path, meta_path=meta_path, subset=subset, test=False, normalize='to_1')  
testset = BBBC(folder_path=folder_path, meta_path=meta_path, subset=subset, test=True, normalize='to_1')


X_train = DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
)
X_test = DataLoader(
    testset,
    batch_size=batch_size,
    shuffle=True,
)

print("sucessfully initialized dataloader")


for i, latent_dim in enumerate(latent_dims):
    name = "L=" + str(latent_dim) + ". "
    print(name)

    VAE_ = VAE(
        latent_dim=latent_dim,
        input_dim=input_dim,
        channels=channels,
    ).to(device)

    if '___ARKIV' in os.listdir():
        print("VAE:")
        summary(VAE_, input_size=(channels, input_dim, input_dim))

    encoder_VAE, decoder_VAE, reconstruction_errors, regularizers, latent_space = VAE_.train_VAE(
        dataloader=X_train, epochs=epochs)
    

    #* Save results

    results_folder = 'results/'
    if not(os.path.exists(results_folder)):
        os.mkdir(results_folder)
     
    plot_elbo(reconstruction_errors, regularizers, name + "ELBO", results_folder=results_folder)
    plot_reconstruction(X_train.dataset, name + "train reconstruction", encoder_VAE, decoder_VAE, latent_dim, channels, input_dim, results_folder=results_folder)
    plot_reconstruction(X_test.dataset, name + "test reconstruction", encoder_VAE, decoder_VAE, latent_dim, channels, input_dim, results_folder=results_folder)
    
    print("sucessfully saved results for " + name)
    
    

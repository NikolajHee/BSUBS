import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
#from torchsummary import summary
from tqdm import tqdm
import os
import sys
from new_net import log_Normal, log_standard_Normal, encoder, decoder, VAE, generate_image, plot_reconstruction


# class to extract features from encoder
class FeatureExtractor(VAE):
    def __init__(self, latent_dim, input_dim, channels, hidden_channels):
        super().__init__(latent_dim, input_dim, channels, hidden_channels)
    
    

    def test_eval(self, dataloader, save_latent=False, results_folder=''):
        # only wors if len of dataloader is divisible by batch_size
        REs, KLs, ELBOs = [], [], []

        moa, compound = [], []

        latent = np.zeros((latent_dim, len(dataloader) * dataloader.batch_size)).T
        self.eval()
        with torch.no_grad():
            for i, batch in tqdm(enumerate(dataloader)):
                x = batch["image"].to(device)

                moa, compound = moa + batch["moa"], compound + batch["compound"]

                if save_latent:
                    elbo, RE, KL, z = self.forward(x, save_latent=save_latent)
                    z = z.detach().numpy()
                    latent[i*dataloader.batch_size:(i+1)*dataloader.batch_size, :] = z

                else:
                    elbo, RE, KL = self.forward(x)

                REs, KLs, ELBOs = REs + [RE.item()], KLs + [KL.item()], ELBOs + [elbo.item()]

            if save_latent: 
                np.savez(results_folder + "latent_space.npz", z=latent, labels=moa, compound=compound)

            return REs, KLs, ELBOs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

latent_dim = 200
epochs = 200
batch_size = 200

input_dim = 68
channels = 3

train_size = 30000
test_size = 1000

epochs, batch_size, train_size = 2, 2, 100

torch.backends.cudnn.deterministic = True
torch.manual_seed(42)
torch.cuda.manual_seed(42)

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


X_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
X_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

VAE_ = FeatureExtractor(
    latent_dim=latent_dim,
    input_dim=input_dim,
    channels=channels,
    hidden_channels=[8,16,32]
).to(device)



#print("VAE:")
#summary(VAE, input_size=(channels, input_dim, input_dim))

encoder_VAE, decoder_VAE, REs, KLs, ELBOs = VAE_.train_VAE(
                dataloader=X_train, epochs=epochs)


results_folder = 'feature_extraction/'
if not(os.path.exists(results_folder)):
    os.mkdir(results_folder)



test_REs, test_KLs, test_ELBOs = VAE_.test_eval(X_test, save_latent=True, results_folder=results_folder)


# torch.save(encoder_VAE, "encoder_VAE.pt")
# torch.save(decoder_VAE, "decoder_VAE.pt")

# np.savez("latent_space_VAE.npz", latent_space=latent_space.detach().numpy())



# save results (elbo)
np.savez(results_folder + 'results.npz', REs=REs, KLs=KLs, ELBOs=ELBOs)

x = np.arange(0, len(REs), 1)

plt.plot(x, REs, label="Reconstruction Error")
plt.plot(x, KLs, label="Regularizer")
plt.plot(x, ELBOs, label="ELBO")
plt.xlabel("iterationns")
plt.title("ELBO Components")
plt.legend()
plt.savefig(results_folder + 'ELBO_components.png')
plt.show()

    

plot_reconstruction(X_train.dataset, 
                        vae = VAE_,
                        name="Training_images_reconstructed", 
                        results_folder=results_folder, 
                        latent_dim=latent_dim, 
                        channels=channels, 
                        input_dim=input_dim)
    

plot_reconstruction(X_test.dataset, 
                    vae = VAE_,
                    name="Test_images_reconstructed", 
                    results_folder=results_folder, 
                    latent_dim=latent_dim, 
                    channels=channels, 
                    input_dim=input_dim)
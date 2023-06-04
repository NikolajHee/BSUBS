import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# from torchsummary import summary
from tqdm import tqdm
import torchvision
from torch.utils.data import Dataset
import pandas as pd
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# detect anomaly
torch.autograd.set_detect_anomaly(True)


class encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, channels):
        super(encoder, self).__init__()
        self.input_dim = input_dim
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=5, padding="same")
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding="same")
        self.fully_connected = nn.Linear(
            32 * self.input_dim * self.input_dim, 2 * latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.view(-1, 32 * self.input_dim * self.input_dim)
        x = self.fully_connected(x)
        return x


class decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, channels, pixel_range):
        super(decoder, self).__init__()
        self.input_dim = input_dim
        self.channels = channels
        self.pixel_range = pixel_range

        self.input = nn.Linear(
            latent_dim, 32 * self.input_dim * self.input_dim)
        self.conv1 = nn.ConvTranspose2d(
            32, 16, kernel_size=5, stride=1, padding=5 - (self.input_dim % 5))
        self.conv2 = nn.ConvTranspose2d(
            16, channels, kernel_size=5, stride=1, padding=5 - (self.input_dim % 5))
        # conversion to [0,1]
        self.softmax = nn.Sigmoid()

    def forward(self, x):
        x = self.input(x)
        x = nn.functional.relu(x)
        x = x.view(-1, 32, self.input_dim, self.input_dim)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.view(-1, self.channels, self.input_dim, self.input_dim)
        # x = self.fully_connected(x)
        # x = x.view(-1, self.channels * self.input_dim *
        #            self.input_dim, self.pixel_range)
        # x = self.softmax(x)
        return x


class VAE(nn.Module):
    def __init__(self, X, pixel_range, latent_dim, input_dim, channels):
        super(VAE, self).__init__()
        self.encoder = encoder(input_dim, latent_dim, channels)
        self.decoder = decoder(input_dim, latent_dim, channels, pixel_range)
        self.input_dim = input_dim
        self.pixel_range = pixel_range
        self.latent_dim = latent_dim

        self.data_length = len(X)
        # self.prior = torch.distributions.MultivariateNormal(loc=torch.zeros(latent_dim), covariance_matrix=torch.eye(latent_dim))

    def encode(self, x):
        mu, log_var = torch.split(
            self.encoder.forward(x), self.latent_dim, dim=1)
        return mu, log_var

    def reparameterization(self, mu, log_var):
        eps = torch.normal(mean=0, std=torch.ones(latent_dim)).to(device)
        return mu + torch.exp(0.5*log_var) * eps

    def decode(self, z):
        return self.decoder.forward(z)

    def forward(self, x, reduction='sum'):
        mu, log_var = self.encode(x)


        # KL divergence term
        p = torch.distributions.Normal(torch.zeros(self.latent_dim).to(device), torch.ones(self.latent_dim).to(device))
        q = torch.distributions.Normal(mu, torch.exp(0.5*log_var))

       
        z = self.reparameterization(mu, log_var)

        log_qzx = q.log_prob(z)
        log_pz = p.log_prob(z)  

        kl = (log_qzx - log_pz) 

        # sum 
        if reduction == 'sum':
            kl = torch.sum(kl, dim=-1)
        else:
            kl = torch.mean(kl, dim=-1)
        
        # reconstruction error
        # x_tilde is the parameters of the distribution

        x_tilde = self.decode(z)


        # this nexxxt part is unclear to me
        log_scale = torch.nn.parameter.Parameter(torch.tensor([0.0]))
        scale = torch.exp(log_scale)
        dist = torch.distributions.Normal(x_tilde, scale.to(device))

        log_pxz = dist.log_prob(x)
        log_pxz = torch.sum(log_pxz, dim=(1, 2, 3))

        reconstruct = -log_pxz


        elbo = kl + reconstruct
        elbo = elbo.mean()
        

        tqdm.write(
            f"ELBO: {elbo}, Reconstruction error: { reconstruct.mean()}, Regularizer: {kl.mean()}")

        return elbo, reconstruct.mean(), kl.mean()

    def train_VAE(self, dataloader, epochs, batch_size, lr=10e-5):
        parameters = [param for param in self.parameters()
                      if param.requires_grad == True]
        optimizer = torch.optim.Adam(parameters, lr=lr)

        reconstruction_errors = []
        regularizers = []

        ## self.train()
        for epoch in tqdm(range(epochs)):
            for batch in tqdm(dataloader):
                
                x = batch['image'].to(device)
                x = x/255
    
                elbo, reconstruction_error, regularizer = self.forward(x)
                reconstruction_errors.append(
                    reconstruction_error.detach().numpy())
                regularizers.append(regularizer.detach().numpy())

                optimizer.zero_grad()
                elbo.backward()
                optimizer.step()

            tqdm.write(
                f"Epoch: {epoch+1}, ELBO: {elbo.detach()}, Reconstruction Error: {reconstruction_error.detach()}, Regularizer: {regularizer.detach()}")

        mu, log_var = self.encode(x)
        latent_space = self.reparameterization(mu, log_var)

        return self.encoder, self.decoder, reconstruction_errors, regularizers, latent_space


def generate_image(X, encoder, decoder, latent_dim, channels, input_dim, batch_size=1):
    encoder.eval()
    decoder.eval()
    X = X.to(device)
    mu, log_var = torch.split(encoder.forward(X/255), latent_dim, dim=1)
    eps = torch.normal(mean=0, std=torch.ones(latent_dim)).to(device)
    z = mu + torch.exp(0.5*log_var) * eps
    x_tilde = decoder.forward(z)
    ## image = torch.argmax(theta, dim=-1)
     #image = image.reshape((batch_size, channels, input_dim, input_dim))
    # image = torch.permute(image, (0, 2, 3, 1))
    log_scale = torch.nn.parameter.Parameter(torch.tensor([0.0]))
    scale = torch.exp(log_scale)
    dist = torch.distributions.Normal(x_tilde, scale.to(device))
    image = dist.sample()
    return image



class BBBC(Dataset):
        def __init__(self, folder_path, meta_path, subset=(390716, 97679), # 1/5 of the data is test data  by default
                                                            test=False):
            self.meta = pd.read_csv(meta_path, index_col=0)
            self.col_names = self.meta.columns
            self.folder_path = folder_path
            self.train_size, self.test_size = subset
            self.test = test    
            self.meta = self.meta.iloc[self.train_size:self.train_size + self.test_size, :] if self.test else self.meta.iloc[:self.train_size,:]
            
        def __len__(self,):
            return self.test_size if self.test else self.train_size
        
        def normalize_to_255(self, x):
            # helper function to normalize to 255
            to_255 = (x/np.max(x)) * 255
            return to_255.astype(np.uint8).reshape((3,68,68))

        def __getitem__(self, idx):
            if self.test: idx += self.train_size
            img_name = os.path.join(self.folder_path,
                                    self.meta[self.col_names[1]][idx], 
                                    self.meta[self.col_names[3]][idx])
            image = np.load(img_name)
            image = self.normalize_to_255(image)
            moa = self.meta[self.col_names[-1]][idx]
            # more relevant stuff here

            sample = {"idx": idx, 
                    "image": image, 
                    "moa": moa, 
                    } # more relevant stuff here

            return sample


latent_dim = 8
epochs = 3
batch_size = 20

pixel_range = 256
input_dim = 68
channels = 3
learning_rate = 1e-5

train_size = 100 # 400000 # #60000
test_size = 10 # 88395 # 10000


# path to singlecells
# main_path = "/Users/nikolaj/Fagprojekt/Data/"
main_path = "/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/"

subset = (train_size, test_size)



dataset_train = BBBC(folder_path=main_path + "singh_cp_pipeline_singlecell_images",
                        meta_path=main_path + "metadata.csv",
                        subset=subset,
                        test=False)

dataset_test = BBBC(folder_path=main_path + "singh_cp_pipeline_singlecell_images",
                        meta_path=main_path + "metadata.csv",
                        subset=subset,
                        test=True)

X_train = DataLoader(dataset_train, batch_size=batch_size)#, shuffle=True, num_workers=0)

X_test = DataLoader(dataset_test, batch_size=batch_size)#, shuffle=True, num_workers=0)  



VAE = VAE(X_train, pixel_range=pixel_range,
        latent_dim=latent_dim, input_dim=input_dim, channels=channels).to(device)


encoder_VAE, decoder_VAE, reconstruction_errors, regularizers, latent_space = VAE.train_VAE(
    dataloader=X_train, epochs=epochs, batch_size=batch_size)


generated_images = []


for batch in X_test:
    for j in range(batch_size):
        image = generate_image(batch['image'][j], encoder_VAE, decoder=decoder_VAE,
                            latent_dim=latent_dim, channels=channels, input_dim=input_dim)
        generated_images.append(image.detach().cpu().numpy())
        if len(generated_images) == 9:
            break
    if len(generated_images) == 9:
            break


# save 
save_folder_path = "VAE_results/"


name = "learning_rate_" + str(learning_rate) + "_latent_dim_" + str(latent_dim) + "_epochs_" + str(epochs) + "_batch_size_" + str(batch_size) + "_"

np.savez(save_folder_path + name + "latent_space.npz", latent_space=latent_space.detach().cpu().numpy())
np.savez(save_folder_path + name + "reconstruction_errors.npz", reconstruction_errors=np.array(reconstruction_errors))
np.savez(save_folder_path + name + "regularizers.npz", regularizers=np.array(regularizers))
np.savez(save_folder_path + name + "generated_images.npz", generated_images=np.array(generated_images))


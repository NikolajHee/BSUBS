import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from torchsummary import summary
import os
import pandas as pd

# TODO: Gridsearch over latent_dim

# TODO: Try with a different posterior distribution


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def log_Categorical(x, theta, num_classes):
    x_one_hot = nn.functional.one_hot(
        x.flatten(start_dim=1, end_dim=-1).long(), num_classes=num_classes)
    log_p = torch.sum(
        x_one_hot * torch.log(theta), dim=-1)
    return log_p


def log_Normal(x, mu, log_var):
    D = x.shape[1]
    log_p = -.5 * ((x - mu) ** 2. * torch.exp(-log_var) +
                   log_var + D * torch.log(2 * torch.tensor(np.pi)))
    return log_p


def log_standard_Normal(x):
    D = x.shape[1]
    log_p = -.5 * (x ** 2. + D * torch.log(2 * torch.tensor(np.pi)))
    return log_p


class encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, channels):
        super(encoder, self).__init__()
        self.input_dim = input_dim
        self.conv1 = nn.Conv2d(channels, 16, kernel_size=5, padding="same")
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding="same")
        self.fully_connected = nn.Linear(
            32 * self.input_dim//2 * self.input_dim//2, 2 * latent_dim)
        # max pooling that halves the input size
        
    def forward(self, x):
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.view(-1, 32 * self.input_dim//2 * self.input_dim//2)
        x = self.fully_connected(x)
        return x


class decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, channels, pixel_range):
        super(decoder, self).__init__()
        self.input_dim = input_dim
        self.channels = channels
        self.pixel_range = pixel_range

        self.input = nn.Linear(
            latent_dim, 32 * self.input_dim//2 * self.input_dim//2)
        self.conv1 = nn.ConvTranspose2d(
            32, 16, kernel_size=5, stride=1, padding=5 - (self.input_dim//2 % 5))
        self.conv2 = nn.ConvTranspose2d(
            16, channels, kernel_size=5, stride=1, padding=5 - (self.input_dim//2 % 5))
        self.fully_connected = nn.Linear(
            channels * self.input_dim//2 * self.input_dim//2, channels * self.input_dim//2 * self.input_dim//2 * pixel_range)
        self.softmax = nn.Softmax(dim=2)  # changed dim from 1 to 2

    def forward(self, x):
        x = self.input(x)
        x = nn.functional.relu(x)
        x = x.view(-1, 32, self.input_dim, self.input_dim)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        x = nn.functional.relu(x)
        x = x.view(-1, self.channels * self.input_dim * self.input_dim)
        x = self.fully_connected(x)
        x = x.view(-1, self.channels * self.input_dim *
                   self.input_dim, self.pixel_range)
        x = self.softmax(x)
        return x


class VAE(nn.Module):
    def __init__(self, X, pixel_range, latent_dim, input_dim, channels):
        super(VAE, self).__init__()
        self.encoder = encoder(input_dim, latent_dim, channels)
        self.decoder = decoder(input_dim, latent_dim, channels, pixel_range)
        self.pixel_range = pixel_range
        self.latent_dim = latent_dim

        self.data_length = len(X)
        self.eps = torch.normal(mean=0, std=torch.ones(latent_dim)).to(device)
        # self.prior = torch.distributions.MultivariateNormal(loc=torch.zeros(latent_dim), covariance_matrix=torch.eye(latent_dim))

    def encode(self, x):
        mu, log_var = torch.split(
            self.encoder.forward(x), self.latent_dim, dim=1)
        return mu, log_var

    def reparameterization(self, mu, log_var):
        return mu + torch.exp(0.5*log_var) * self.eps

    def decode(self, z):
        return self.decoder.forward(z)

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterization(mu, log_var)

        theta = self.decode(z)

        log_posterior = log_Normal(z, mu, log_var)
        log_prior = log_standard_Normal(z)
        log_like = log_Categorical(x, theta, self.pixel_range)

        reconstruction_error = - torch.sum(log_like, dim=-1).mean()
        regularizer = - torch.sum(log_prior - log_posterior, dim=-1).mean()

        elbo = reconstruction_error + regularizer

        tqdm.write(
            f"ELBO: {elbo.detach()}, Reconstruction error: {reconstruction_error.detach()}, Regularizer: {regularizer.detach()}")

        return elbo, reconstruction_error, regularizer

    def train_VAE(self, dataloader, epochs, batch_size, lr=10e-5):
        parameters = [param for param in self.parameters()
                      if param.requires_grad == True]
        optimizer = torch.optim.Adam(parameters, lr=lr)

        reconstruction_errors = []
        regularizers = []

        self.train()
        for epoch in tqdm(range(epochs)):
            for batch in tqdm(dataloader):
                x = batch.to(device)
                optimizer.zero_grad()
                elbo, reconstruction_error, regularizer = self.forward(x)
                reconstruction_errors.append(
                    reconstruction_error.detach().numpy())
                regularizers.append(regularizer.detach().numpy())
                elbo.backward(retain_graph=True)
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
    mu, log_var = torch.split(encoder.forward(X), latent_dim, dim=1)
    eps = torch.normal(mean=0, std=torch.ones(latent_dim))
    z = mu + torch.exp(0.5*log_var) * eps
    theta = decoder.forward(z)
    image = torch.argmax(theta, dim=-1)
    image = image.reshape((batch_size, channels, input_dim, input_dim))
    image = torch.permute(image, (0, 2, 3, 1))
    image = image.numpy()
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



latent_dim = 20
epochs = 5
batch_size = 10

pixel_range = 256
input_dim = 68
channels = 3

train_size = 100
test_size = 100

# path to singlecells
main_path = "/Users/nikolaj/Fagprojekt/Data/"

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

print("VAE:")
summary(VAE, input_size=(channels, input_dim, input_dim))

encoder_VAE, decoder_VAE, reconstruction_errors, regularizers, latent_space = VAE.train_VAE(
    dataloader=X_train, epochs=epochs, batch_size=batch_size)



#  torch.save(encoder_VAE, "encoder_VAE.pt")
# torch.save(decoder_VAE, "decoder_VAE.pt")

# np.savez("latent_space_VAE.npz", latent_space=latent_space.detach().numpy())

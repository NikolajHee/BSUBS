import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchsummary import summary
from tqdm import tqdm
from torch.nn import functional as F
import os


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_Normal(x, mu, log_var):
    D = x.shape[1]
    log_p = -0.5 * (
        (x - mu) ** 2.0 * torch.exp(-log_var)
        + log_var
        + D * torch.log(2 * torch.tensor(np.pi))
    )
    return log_p


def log_standard_Normal(x):
    D = x.shape[1]
    log_p = -0.5 * (x**2.0 + D * torch.log(2 * torch.tensor(np.pi)))
    return log_p


class encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, channels):
        super(encoder, self).__init__()
        self.input_dim = input_dim

        self.conv1 = nn.Conv2d(channels, 16, kernel_size=5, padding="same")
        self.fully_connected = nn.Linear(
            16 * self.input_dim * self.input_dim, 2 * latent_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.LeakyReLU(0.01)(x)
        x = x.view(-1, 16 * self.input_dim * self.input_dim)
        x = self.fully_connected(x)
        x = nn.LeakyReLU(0.01)(x)
        return x


class decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, channels):
        super(decoder, self).__init__()
        self.input_dim = input_dim
        self.channels = channels

        self.input = nn.Linear(
            latent_dim, 16 * self.input_dim * self.input_dim)
        self.conv2 = nn.ConvTranspose2d(
            16, channels, kernel_size=5, stride=1, padding=5 - (self.input_dim % 5))
        self.output = nn.Linear(channels * self.input_dim * self.input_dim,
                                2 * channels * self.input_dim * self.input_dim)

        # self.softmax = nn.Softmax(dim=1)
        # self.softmax = nn.Sigmoid()

    def forward(self, x):
        x = self.input(x)
        x = nn.LeakyReLU(0.01)(x)
        x = x.view(-1, 16, self.input_dim, self.input_dim)
        x = self.conv2(x)
        x = nn.LeakyReLU(0.01)(x)
        x = x.view(-1, self.channels * self.input_dim * self.input_dim)
        x = self.output(x)
        x = nn.LeakyReLU(0.01)(x)
        # x = self.softmax(x) # i dont think this is needed.. but maybe?
        return x


class VAE(nn.Module):
    def __init__(self, latent_dim, input_dim, channels):
        super(VAE, self).__init__()
        self.encoder = encoder(input_dim, latent_dim, channels)
        self.decoder = decoder(input_dim, latent_dim, channels)
        self.latent_dim = latent_dim
        self.channels = channels
        self.input_dim = input_dim

    def encode(self, x):
        mu, log_var = torch.split(
            self.encoder.forward(x), self.latent_dim, dim=1)
        return mu, log_var

    def reparameterization(self, mu, log_var):
        eps = torch.normal(mean=0, std=torch.ones(self.latent_dim)).to(device)
        return mu + torch.exp(0.5 * log_var) * eps

    def decode(self, z):
        mu, log_var = torch.split(
            self.decoder.forward(z), self.channels * self.input_dim * self.input_dim, dim=1)
        std = torch.exp(0.5 * log_var)
        return mu, std

    def forward(self, x, print_=True):
        mu, log_var = self.encode(x/255)
        z = self.reparameterization(mu, log_var)

        decode_mu, decode_std = self.decode(z)
        recon_x = torch.normal(mean=decode_mu, std=decode_std)

        log_posterior = log_Normal(z, mu, log_var)
        log_prior = log_standard_Normal(z)

        # initial try
        # log_like = torch.normal(mean = mean, std = 0.001)
        # log_like = torch.distributions.LogNormal(loc=mean, scale=0.01)
        # recon_x = log_like.sample()

        # this works? like some papers say to do this
        reconstruction_error = F.mse_loss(
            recon_x, x.view(-1, self.channels * self.input_dim * self.input_dim)/255, reduction="none").sum(-1)
        reconstruction_error = reconstruction_error.to(device)

        regularizer = -torch.sum(log_prior - log_posterior, dim=-1)
        # regularizer  = -0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1) # alternative

        elbo = (reconstruction_error + regularizer).mean()

        # print(f"ELBO: {elbo.detach()}, Reconstruction error: {reconstruction_error.detach().mean()}, Regularizer: {regularizer.detach().mean()}")
        if print_:
            tqdm.write(
                f"ELBO: {elbo.detach()}, Reconstruction error: {reconstruction_error.detach().mean()}, Regularizer: {regularizer.detach().mean()}")

        return elbo, reconstruction_error.mean(), regularizer.mean()

    def initialise(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.sparse_(m.weight, sparsity=1)
                if m.bias is not None:
                    m.bias.data.fill_(3)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.dirac_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0)

        self.apply(_init_weights)

    def train_VAE(self, dataloader, epochs, lr=1e-5):
        parameters = [
            param for param in self.parameters() if param.requires_grad == True
        ]
        optimizer = torch.optim.Adam(parameters, lr=lr)

        reconstruction_errors = []
        regularizers = []

        # self.initialise()
        self.train()
        for epoch in tqdm(range(epochs)):
            for batch in tqdm(dataloader):
                x = batch.to(device)
                optimizer.zero_grad()
                elbo, reconstruction_error, regularizer = self.forward(x)
                reconstruction_errors.append(
                    reconstruction_error.cpu().detach().numpy())
                regularizers.append(regularizer.detach().numpy())
                elbo.backward(retain_graph=True)
                optimizer.step()

            tqdm.write(
                f"Epoch: {epoch+1}, ELBO: {elbo.detach()}, Reconstruction Error: {reconstruction_error.detach()}, Regularizer: {regularizer.detach()}"
            )

        mu, log_var = self.encode(x)
        latent_space = self.reparameterization(mu, log_var)

        return (
            self.encoder,
            self.decoder,
            reconstruction_errors,
            regularizers,
            latent_space,
        )


def generate_image(X, encoder, decoder, latent_dim, channels, input_dim, batch_size=1):
    encoder.eval()
    decoder.eval()
    X = X.to(device)
    mu, log_var = torch.split(encoder.forward(X), latent_dim, dim=1)
    eps = torch.normal(mean=0, std=torch.ones(latent_dim)).to(device)
    z = mu + torch.exp(0.5 * log_var) * eps
    mean = decoder.forward(z)

    log_like = torch.normal(mean=mean, std=0.1).to(device)
    recon_x = log_like
    image = recon_x.reshape((channels, input_dim, input_dim))
    image = image.cpu().detach().numpy()
    return image